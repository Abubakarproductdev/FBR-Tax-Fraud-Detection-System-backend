"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4: Heterogeneous GNN & Hybrid Anomaly Engine (Memgraph Edition)       ║
║                                                                              ║
║  Connects to Memgraph, executes MAGE algorithms (Louvain/PageRank),          ║
║  extracts explicit topological features for Explainable AI, trains an        ║
║  Isolation Forest, and calculates a hybrid defensible risk score.            ║
║                                                                              ║
║  Author : Lead Data Engineer                                                 ║
║  Inputs : Memgraph Graph, layer3_financial_profiles.csv                      ║
║  Output : layer4_hybrid_profiles.csv                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ── Memgraph Cloud Credentials ──
URI = os.getenv("MEMGRAPH_URI")
USER = os.getenv("MEMGRAPH_USER")
PASSWORD = os.getenv("MEMGRAPH_PASSWORD")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def execute_mage_algorithms(session):
    print("\n[2/5] Step A: Executing Memgraph MAGE Analytics (In-Database)...")
    
    # Run Louvain
    print("       -> Running Louvain Modularity...")
    session.run("CALL community_detection.get() YIELD node, community_id SET node.community_id = community_id")
    
    # Run PageRank
    print("       -> Running PageRank...")
    session.run("CALL pagerank.get() YIELD node, rank SET node.wealth_centrality_score = rank")
    
    print("       ✓ MAGE graph metrics successfully written to Memgraph.")

    
def extract_features(session):
    # Extract explicit network structure metrics using Memgraph-safe Cypher syntax
    query = """
    MATCH (p:Person)
    
    // Count total connections safely
    OPTIONAL MATCH (p)-[r]-()
    WITH p, count(r) AS total_network_degree
    
    // Count proxy connections safely
    OPTIONAL MATCH (p)-[r2:CO_LOCATED_WITH]-()
    WITH p, total_network_degree, count(r2) AS proxy_network_size
    
    // Return final metrics
    RETURN p.canonical_id AS canonical_id, 
           COALESCE(p.community_id, 0) AS community_id,
           COALESCE(p.wealth_centrality_score, 0) AS wealth_centrality_score,
           total_network_degree,
           proxy_network_size
    """
    result = session.run(query)
    records = [record.data() for record in result]
    return pd.DataFrame(records)

def main():
    print("="*78)
    print("  LAYER 4 ── Heterogeneous MAGE & Hybrid Anomaly Engine")
    print("="*78)

    # 1. Connect to Memgraph
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("\n[1/5] ✓ Connected to Memgraph Cloud via neo4j driver.")
    except Exception as e:
        print(f"\n❌ Error connecting to Memgraph: {e}")
        return

    try:
        with driver.session() as session:
            # 2. Step A: MAGE Cypher Calls
            execute_mage_algorithms(session)

            # 3. Step B: Unsupervised Anomaly Detection
            print("\n[3/5] Step B: Unsupervised Anomaly Detection (Isolation Forest)...")
            df_gds = extract_features(session)
            
            if df_gds.empty:
                print("       ❌ No Person nodes found.")
                return
            
            print(f"       -> Extracted features for {len(df_gds):,} individuals.")

            # Create feature matrix using structural topology instead of embeddings
            features = df_gds[['wealth_centrality_score', 'total_network_degree', 'proxy_network_size']].values
            
            # Train Isolation Forest
            clf = IsolationForest(contamination=0.15, random_state=42)
            clf.fit(features)
            
            # Get anomaly scores
            raw_scores = clf.score_samples(features)
            inverted_scores = -raw_scores # Higher is more anomalous
            
            # Convert to 0-100 scale
            scaler = MinMaxScaler(feature_range=(0, 100))
            df_gds['gds_structural_anomaly_score'] = scaler.fit_transform(inverted_scores.reshape(-1, 1)).flatten()
            
            # Normalize PageRank to 0-100
            df_gds['normalized_pagerank'] = scaler.fit_transform(df_gds[['wealth_centrality_score']]).flatten()
            
            print("       ✓ Isolation Forest trained and structural anomaly scores computed.")

            # 4. Step C: The Hybrid Defensible Score
            print("\n[4/5] Step C: Calculating Hybrid Defensible Score...")
            
            layer3_path = os.path.join(BASE_DIR, "layer3_financial_profiles.csv")
            if not os.path.exists(layer3_path):
                print(f"       ❌ Missing input file: {layer3_path}")
                return
                
            df_l3 = pd.read_csv(layer3_path, dtype=str)
            
            if 'deviation_score' in df_l3.columns:
                df_l3['deviation_score'] = pd.to_numeric(df_l3['deviation_score'], errors='coerce').fillna(0)
            else:
                df_l3['deviation_score'] = 0.0

            df_merged = df_l3.merge(df_gds, on="canonical_id", how="left")
            
            df_merged['gds_structural_anomaly_score'] = df_merged['gds_structural_anomaly_score'].fillna(0)
            df_merged['normalized_pagerank'] = df_merged['normalized_pagerank'].fillna(0)
            
            df_merged['final_hybrid_risk_score'] = (
                (0.5 * df_merged['deviation_score']) + 
                (0.3 * df_merged['gds_structural_anomaly_score']) + 
                (0.2 * df_merged['normalized_pagerank'])
            )
            
            df_merged = df_merged.sort_values('final_hybrid_risk_score', ascending=False)
            output_path = os.path.join(BASE_DIR, "layer4_hybrid_profiles.csv")
            df_merged.to_csv(output_path, index=False)
            
            print(f"       ✓ Exported Layer 4 hybrid profiles to: {output_path}")

            # 5. Output Top 5
            print("\n[5/5] Top 5 Highest-Risk Individuals Identified:")
            print("-" * 110)
            
            top_5 = df_merged.head(5)
            has_name = 'normalized_name' in top_5.columns
            
            header = f"{'Rank':<5} | {'Canonical ID':<15} | {'Name' if has_name else 'N/A':<20} | {'Dev Score':<10} | {'Struct Score':<12} | {'Norm PageRank':<13} | {'Hybrid Score':<12}"
            print(header)
            print("-" * 110)
            
            for idx, row in enumerate(top_5.itertuples(), 1):
                c_id = getattr(row, 'canonical_id', 'N/A')
                name = getattr(row, 'normalized_name', 'N/A')[:18] if has_name else 'N/A'
                dev = getattr(row, 'deviation_score', 0)
                struct = getattr(row, 'gds_structural_anomaly_score', 0)
                pr = getattr(row, 'normalized_pagerank', 0)
                hybrid = getattr(row, 'final_hybrid_risk_score', 0)
                
                print(f"{idx:<5} | {c_id:<15} | {name:<20} | {dev:10.2f} | {struct:12.2f} | {pr:13.2f} | {hybrid:12.2f}")
            
            print("-" * 110)
            print("\n" + "="*78)
            print("  Pipeline execution completed.")
            print("="*78)
            
    finally:
        driver.close()

if __name__ == "__main__":
    main()