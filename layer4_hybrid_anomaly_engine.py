"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4: Heterogeneous GNN & Hybrid Anomaly Engine                        ║
║                                                                            ║
║  Extracts the Neo4j graph into a PyTorch Geometric HeteroData object.      ║
║  Trains an unsupervised Heterogeneous Graph Autoencoder to compute         ║
║  node-level structural reconstruction losses. Merges this structural       ║
║  anomaly score with Layer 3's financial deviation score to produce         ║
║  a highly defensible Hybrid Risk Score.                                    ║
║                                                                            ║
║  Author : Senior AI Researcher                                             ║
║  Inputs : Neo4j Database, layer3_financial_profiles.csv                    ║
║  Output : layer4_hybrid_profiles.csv                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# § 1. DATA EXTRACTION & HETERODATA CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_data_from_neo4j():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    data_dict = {}
    
    with driver.session() as session:
        # Nodes
        print("       Fetching Person nodes...")
        persons = session.run("MATCH (n:Person) RETURN n.canonical_id AS id").data()
        data_dict['Person'] = {'ids': [r['id'] for r in persons]}
        
        print("       Fetching Vehicle nodes...")
        vehicles = session.run("MATCH (n:Vehicle) RETURN n.vehicle_reg_no AS id, toFloat(n.engine_capacity_cc) AS f1").data()
        data_dict['Vehicle'] = {
            'ids': [r['id'] for r in vehicles],
            'features': [[r['f1'] or 0.0] for r in vehicles]
        }
        
        print("       Fetching Property nodes...")
        props = session.run("MATCH (n:Property) RETURN n.registry_deed_no AS id, toFloat(n.property_value_pkr) AS f1").data()
        data_dict['Property'] = {
            'ids': [r['id'] for r in props],
            'features': [[r['f1'] or 0.0] for r in props]
        }
        
        print("       Fetching Meter nodes...")
        meters = session.run("MATCH (n:Meter) RETURN n.meter_ref_no AS id, toFloat(n.avg_monthly_bill_pkr) AS f1").data()
        data_dict['Meter'] = {
            'ids': [r['id'] for r in meters],
            'features': [[r['f1'] or 0.0] for r in meters]
        }
        
        print("       Fetching TaxReturn nodes...")
        tax = session.run("MATCH (n:TaxReturn) RETURN n.fbr_id AS id, toFloat(n.declared_income_pkr) AS f1, toFloat(n.tax_paid_pkr) AS f2").data()
        data_dict['TaxReturn'] = {
            'ids': [r['id'] for r in tax],
            'features': [[r['f1'] or 0.0, r['f2'] or 0.0] for r in tax]
        }

        # Edges
        print("       Fetching Edges...")
        data_dict['Edges'] = {
            'OWNS': session.run("MATCH (p:Person)-[:OWNS]->(v:Vehicle) RETURN p.canonical_id AS src, v.vehicle_reg_no AS dst").data(),
            'RESIDES_AT': session.run("MATCH (p:Person)-[:RESIDES_AT]->(pr:Property) RETURN p.canonical_id AS src, pr.registry_deed_no AS dst").data(),
            'CONSUMES_VIA': session.run("MATCH (p:Person)-[:CONSUMES_VIA]->(m:Meter) RETURN p.canonical_id AS src, m.meter_ref_no AS dst").data(),
            'FILED': session.run("MATCH (p:Person)-[:FILED]->(t:TaxReturn) RETURN p.canonical_id AS src, t.fbr_id AS dst").data()
        }

    driver.close()
    return data_dict

def build_heterodata(data_dict):
    data = HeteroData()
    scaler = StandardScaler()
    
    # ID Mappings (str -> int)
    mappings = {}
    
    for node_type in ['Person', 'Vehicle', 'Property', 'Meter', 'TaxReturn']:
        ids = data_dict[node_type]['ids']
        mappings[node_type] = {name: i for i, name in enumerate(ids)}
        
        if node_type == 'Person':
            # Person has no natural features in Graph, initialize with ones
            x = torch.ones((len(ids), 1), dtype=torch.float)
        else:
            feats = np.array(data_dict[node_type]['features'])
            if len(feats) > 0:
                feats = scaler.fit_transform(feats)
            else:
                feats = np.empty((0, 1)) # handle empty
            x = torch.tensor(feats, dtype=torch.float)
        data[node_type].x = x

    # Edges
    edge_types = [
        ('Person', 'OWNS', 'Vehicle'),
        ('Person', 'RESIDES_AT', 'Property'),
        ('Person', 'CONSUMES_VIA', 'Meter'),
        ('Person', 'FILED', 'TaxReturn')
    ]
    
    for src_type, rel_type, dst_type in edge_types:
        edges = data_dict['Edges'][rel_type]
        src_mapping = mappings[src_type]
        dst_mapping = mappings[dst_type]
        
        src_indices = [src_mapping[e['src']] for e in edges if e['src'] in src_mapping and e['dst'] in dst_mapping]
        dst_indices = [dst_mapping[e['dst']] for e in edges if e['src'] in src_mapping and e['dst'] in dst_mapping]
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data[src_type, rel_type, dst_type].edge_index = edge_index

    # Add reverse edges to allow bidirectional message passing
    data = T.ToUndirected()(data)
    
    return data, mappings['Person']

# ──────────────────────────────────────────────────────────────────────────────
# § 2. GNN MODEL (AUTOENCODER)
# ──────────────────────────────────────────────────────────────────────────────

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def forward(self, z_dict, edge_label_index, src_type, dst_type):
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return (z_src * z_dst).sum(dim=-1)

class HeteroAutoEncoder(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        # We use to_hetero to automatically convert a homogeneous GNN to a hetero GNN
        self.encoder = to_hetero(GNNEncoder(hidden_channels, out_channels), metadata, aggr='sum')
        self.decoder = EdgeDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index, src_type, dst_type):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index, src_type, dst_type)

# ──────────────────────────────────────────────────────────────────────────────
# § 3. TRAINING & ANOMALY SCORING
# ──────────────────────────────────────────────────────────────────────────────

def generate_negative_edges(edge_index, num_src_nodes, num_dst_nodes):
    # Generates random negative edges for BCE loss
    num_neg_edges = edge_index.size(1)
    neg_src = torch.randint(0, num_src_nodes, (num_neg_edges,))
    neg_dst = torch.randint(0, num_dst_nodes, (num_neg_edges,))
    return torch.stack([neg_src, neg_dst], dim=0)

def main():
    print("="*78)
    print("  LAYER 4 ── Heterogeneous GNN & Hybrid Anomaly Engine")
    print("="*78)

    # 1. Load Data
    print("\n[1/5] Extracting Graph from Neo4j & Constructing HeteroData...")
    data_dict = fetch_data_from_neo4j()
    data, person_mapping = build_heterodata(data_dict)
    print(f"       ✓ Graph built successfully: {data}")

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = HeteroAutoEncoder(data.metadata(), hidden_channels=32, out_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Edge types to reconstruct (forward edges only for loss calculation)
    target_edge_types = [
        ('Person', 'OWNS', 'Vehicle'),
        ('Person', 'RESIDES_AT', 'Property'),
        ('Person', 'CONSUMES_VIA', 'Meter'),
        ('Person', 'FILED', 'TaxReturn')
    ]

    # 3. Train
    print("\n[2/5] Training Unsupervised Heterogeneous Graph Autoencoder...")
    model.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        total_loss = 0
        
        for edge_type in target_edge_types:
            src_type, _, dst_type = edge_type
            
            # Positive edges
            pos_edge_index = data[edge_type].edge_index
            if pos_edge_index.size(1) == 0:
                continue
                
            pos_pred = model(data.x_dict, data.edge_index_dict, pos_edge_index, src_type, dst_type)
            
            # Negative edges
            neg_edge_index = generate_negative_edges(
                pos_edge_index, 
                data[src_type].num_nodes, 
                data[dst_type].num_nodes
            ).to(device)
            neg_pred = model(data.x_dict, data.edge_index_dict, neg_edge_index, src_type, dst_type)
            
            # Labels (1 for positive, 0 for negative)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
            
            loss = F.binary_cross_entropy_with_logits(pred, target)
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"       Epoch {epoch:>3}/150 | Loss: {total_loss.item():.4f}")

    print("       ✓ Model training complete.")

    # 4. Anomaly Scoring
    print("\n[3/5] Calculating Node-Level Structural Anomaly Scores...")
    model.eval()
    
    # We compute BCE loss per positive edge to find how poorly the model reconstructs it
    # Higher loss = structure anomalous compared to the rest of the graph
    
    person_losses = torch.zeros(data['Person'].num_nodes, device=device)
    person_edge_counts = torch.zeros(data['Person'].num_nodes, device=device)

    with torch.no_grad():
        for edge_type in target_edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
                
            # Logits for positive edges
            logits = model(data.x_dict, data.edge_index_dict, edge_index, src_type, dst_type)
            # BCE loss per edge (target is 1)
            edge_losses = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
            
            # Accumulate loss to the source Person node
            src_nodes = edge_index[0]
            person_losses.scatter_add_(0, src_nodes, edge_losses)
            
            # Count edges per person
            ones = torch.ones_like(src_nodes, dtype=torch.float)
            person_edge_counts.scatter_add_(0, src_nodes, ones)

    # Avoid division by zero
    person_edge_counts[person_edge_counts == 0] = 1.0
    avg_person_losses = (person_losses / person_edge_counts).cpu().numpy().reshape(-1, 1)

    # Normalize to 0-100 score using MinMax
    scaler = MinMaxScaler((0, 100))
    gnn_scores = scaler.fit_transform(avg_person_losses).flatten()
    print("       ✓ Extracted structural anomaly scores (0-100).")

    # 5. The Hybrid Defensible Score
    print("\n[4/5] Merging with Layer 3 & Computing Final Hybrid Risk Score...")
    l3_path = os.path.join(BASE_DIR, "layer3_financial_profiles.csv")
    out_path = os.path.join(BASE_DIR, "layer4_hybrid_profiles.csv")
    
    df = pd.read_csv(l3_path)
    
    # Create mapping array
    inv_person_mapping = {v: k for k, v in person_mapping.items()}
    
    # Attach GNN score
    score_map = {inv_person_mapping[i]: float(gnn_scores[i]) for i in range(len(gnn_scores))}
    df['gnn_structural_anomaly_score'] = df['canonical_id'].map(score_map).fillna(0.0).round(2)
    
    # Calculate Hybrid Score: 60% Financial Deviation, 40% Structural Anomaly
    df['final_hybrid_risk_score'] = ((0.6 * df['deviation_score']) + (0.4 * df['gnn_structural_anomaly_score'])).round(2)
    
    # Sort
    df.sort_values('final_hybrid_risk_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 6. Export
    df.to_csv(out_path, index=False)
    print(f"       ✓ Saved updated profiles to {out_path}")

    print("\n" + "═" * 78)
    print("  TOP 5 HIGHEST-RISK HYBRID PROFILES (Layer 4)")
    print("═" * 78)

    top5 = df.head(5)
    for idx, row in top5.iterrows():
        print(f"\n  ┌─ #{idx + 1}  canonical_id: {row['canonical_id']}")
        print(f"  │   Declared Income:       PKR {row['total_declared_income']:>15,.0f}")
        print(f"  │   Layer 3 Deviation:     {row['deviation_score']:>8.2f} / 100")
        print(f"  │   Layer 4 GNN Anomaly:   {row['gnn_structural_anomaly_score']:>8.2f} / 100")
        print(f"  └─  FINAL HYBRID SCORE:    {row['final_hybrid_risk_score']:>8.2f} / 100")

    print("\n" + "─" * 78)
    print("  Layer 4 pipeline complete.\n")


if __name__ == "__main__":
    main()
