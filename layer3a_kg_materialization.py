"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3A: Knowledge Graph Materialization (Neo4j)                         ║
║                                                                            ║
║  Connects to a Neo4j AuraDB instance, drops existing data, sets up         ║
║  constraints, and ingests canonical entities along with their assets       ║
║  and utility links as a unified knowledge graph.                           ║
║                                                                            ║
║  Author : Senior Graph Database Architect                                  ║
║  Inputs : layer2_resolved_entities.csv, fbr_returns.csv,                   ║
║           excise_vehicles.csv, property_transactions.csv,                  ║
║           utility_bills.csv                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ── Memgraph Cloud Credentials ──
URI = os.getenv("MEMGRAPH_URI")
USER = os.getenv("MEMGRAPH_USER")
PASSWORD = os.getenv("MEMGRAPH_PASSWORD")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def drop_all_data(session):
    print("\n[1/7] Dropping all existing data (clean slate)...")
    # For small/medium datasets, a direct DETACH DELETE works well.
    session.run("MATCH (n) DETACH DELETE n")
    print("       ✓ All existing nodes and relationships dropped.")

def create_constraints(session):
    print("\n[2/7] Creating uniqueness constraints...")
    
    constraints = [
        ("person_unique", "Person", "canonical_id"),
        ("vehicle_unique", "Vehicle", "vehicle_reg_no"),
        ("property_unique", "Property", "registry_deed_no"),
        ("meter_unique", "Meter", "meter_ref_no"),
        ("taxreturn_unique", "TaxReturn", "fbr_id")
    ]
    
    for name, label, prop in constraints:
        try:
            session.run(f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
            print(f"       ✓ Constraint created: (:{label} {{{prop}: UNIQUE}})")
        except Exception as e:
            print(f"       - Failed to create constraint {name}: {e}")

def ingest_persons(session):
    print("\n[3/7] Ingesting Persons from layer2_resolved_entities.csv...")
    df = pd.read_csv(os.path.join(BASE_DIR, "layer2_resolved_entities.csv"), dtype=str).fillna("")
    
    # Drop duplicates to ensure we only have one Person node per canonical_id
    unique_persons = df.drop_duplicates(subset=["canonical_id"])
    records = unique_persons.to_dict("records")
    
    query = """
    UNWIND $batch AS row
    MERGE (p:Person {canonical_id: row.canonical_id})
    SET p.normalized_name = row.normalized_name,
        p.normalized_address = row.normalized_address
    """
    
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        session.run(query, batch=batch)
    
    print(f"       ✓ Ingested {len(records):,} unique Person nodes.")

def ingest_vehicles(session, layer2_df):
    print("\n[4/7] Ingesting Vehicles & OWNS relationships...")
    df = pd.read_csv(os.path.join(BASE_DIR, "excise_vehicles.csv"), dtype=str).fillna("")
    
    l2_excise = layer2_df[layer2_df['source_database'].str.upper() == 'EXCISE']
    merged = df.merge(l2_excise, left_on='vehicle_reg_no', right_on='source_record_id', how='inner')
    records = merged.to_dict("records")
    
    query = """
    UNWIND $batch AS row
    MERGE (v:Vehicle {vehicle_reg_no: row.vehicle_reg_no})
    SET v.engine_capacity_cc = row.engine_capacity_cc
    WITH v, row
    MATCH (p:Person {canonical_id: row.canonical_id})
    MERGE (p)-[:OWNS]->(v)
    """
    
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        session.run(query, batch=batch)
        
    print(f"       ✓ Created {len(records):,} Vehicles and (Person)-[:OWNS]->(Vehicle) relationships.")

def ingest_properties(session, layer2_df):
    print("\n[5/7] Ingesting Properties & RESIDES_AT relationships...")
    df = pd.read_csv(os.path.join(BASE_DIR, "property_transactions.csv"), dtype=str).fillna("")
    
    l2_prop = layer2_df[layer2_df['source_database'].str.upper() == 'PROPERTY']
    merged = df.merge(l2_prop, left_on='registry_deed_no', right_on='source_record_id', how='inner')
    records = merged.to_dict("records")
    
    query = """
    UNWIND $batch AS row
    MERGE (prop:Property {registry_deed_no: row.registry_deed_no})
    SET prop.area_marla = row.area_marla,
        prop.property_value_pkr = row.property_value_pkr
    WITH prop, row
    MATCH (p:Person {canonical_id: row.canonical_id})
    MERGE (p)-[:RESIDES_AT]->(prop)
    """
    
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        session.run(query, batch=batch)
        
    print(f"       ✓ Created {len(records):,} Properties and (Person)-[:RESIDES_AT]->(Property) relationships.")

def ingest_utilities(session, layer2_df):
    print("\n[6/7] Ingesting Meters & CONSUMES_VIA relationships...")
    df = pd.read_csv(os.path.join(BASE_DIR, "utility_bills.csv"), dtype=str).fillna("")
    
    l2_util = layer2_df[layer2_df['source_database'].str.upper() == 'DISCO']
    merged = df.merge(l2_util, left_on='meter_ref_no', right_on='source_record_id', how='inner')
    records = merged.to_dict("records")
    
    query = """
    UNWIND $batch AS row
    MERGE (m:Meter {meter_ref_no: row.meter_ref_no})
    SET m.avg_monthly_bill_pkr = row.avg_monthly_bill_pkr,
        m.provider_type = row.provider_type
    WITH m, row
    MATCH (p:Person {canonical_id: row.canonical_id})
    MERGE (p)-[:CONSUMES_VIA]->(m)
    """
    
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        session.run(query, batch=batch)
        
    print(f"       ✓ Created {len(records):,} Meters and (Person)-[:CONSUMES_VIA]->(Meter) relationships.")

def ingest_fbr_returns(session, layer2_df):
    print("\n[7/7] Ingesting TaxReturns & FILED relationships...")
    df = pd.read_csv(os.path.join(BASE_DIR, "fbr_returns.csv"), dtype=str).fillna("")
    
    l2_fbr = layer2_df[layer2_df['source_database'].str.upper() == 'FBR']
    merged = df.merge(l2_fbr, left_on='fbr_id', right_on='source_record_id', how='inner')
    records = merged.to_dict("records")
    
    query = """
    UNWIND $batch AS row
    MERGE (t:TaxReturn {fbr_id: row.fbr_id})
    SET t.declared_income_pkr = row.declared_income_pkr,
        t.tax_paid_pkr = row.tax_paid_pkr,
        t.fiscal_year = row.fiscal_year
    WITH t, row
    MATCH (p:Person {canonical_id: row.canonical_id})
    MERGE (p)-[:FILED]->(t)
    """
    
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        session.run(query, batch=batch)
        
    print(f"       ✓ Created {len(records):,} TaxReturns and (Person)-[:FILED]->(TaxReturn) relationships.")

def infer_spatial_edges(session):
    print("\n[8/8] Inferring Spatial/Behavioral Edges...")
    
    query_colocated = """
    MATCH (p1:Person), (p2:Person)
    WHERE p1.canonical_id < p2.canonical_id 
      AND p1.normalized_address = p2.normalized_address 
      AND p1.normalized_address <> ""
    MERGE (p1)-[:CO_LOCATED_WITH]->(p2)
    MERGE (p2)-[:CO_LOCATED_WITH]->(p1)
    """
    session.run(query_colocated)
    print("       ✓ Created [:CO_LOCATED_WITH] edges for persons sharing addresses.")
    
    query_wealth = """
    MATCH (p1:Person)-[:CO_LOCATED_WITH]-(p2:Person)-[:RESIDES_AT]->(prop:Property)
    WHERE NOT (p1)-[:RESIDES_AT]->(prop)
    MERGE (p1)-[:SHARED_HOUSEHOLD_WEALTH]->(prop)
    """
    session.run(query_wealth)
    print("       ✓ Created [:SHARED_HOUSEHOLD_WEALTH] inferred edges.")

def main():
    print("="*78)
    print("  LAYER 3A ── Knowledge Graph Materialization (Memgraph)")
    print("="*78)
    
    # Load layer2 once to map source IDs to canonical IDs for relationships
    layer2_df = pd.read_csv(os.path.join(BASE_DIR, "layer2_resolved_entities.csv"), dtype=str).fillna("")
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        # Verify connectivity
        driver.verify_connectivity()
        print("\n✓ Successfully connected to Memgraph Cloud instance.")
        
        with driver.session() as session:
            drop_all_data(session)
            create_constraints(session)
            
            ingest_persons(session)
            ingest_vehicles(session, layer2_df)
            ingest_properties(session, layer2_df)
            ingest_utilities(session, layer2_df)
            ingest_fbr_returns(session, layer2_df)
            infer_spatial_edges(session)
            
        driver.close()
        print("\n" + "="*78)
        print("  Graph materialization successfully completed!")
        print("="*78)
    except Exception as e:
        print(f"\n❌ Error connecting to or updating Memgraph: {e}")

if __name__ == "__main__":
    main()
