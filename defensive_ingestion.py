import duckdb
import time
import logging

def build_ingestion_layer():
    print("Starting Defensive Data Ingestion Layer...")
    start_time = time.time()
    
    # Connect to an in-memory DuckDB database
    print("Connecting to in-memory DuckDB database...")
    con = duckdb.connect(database=':memory:')

    query = """
    COPY (
        -- FBR Returns
        SELECT
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(taxpayer_name)), '\\s+', ' ', 'g'), '') AS normalized_name,
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(reported_address)), '\\s+', ' ', 'g'), '') AS normalized_address,
            fbr_id AS source_record_id,
            'FBR' AS source_database
        FROM read_csv('fbr_returns.csv', all_varchar=true)

        UNION ALL

        -- Excise Vehicles
        SELECT
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(owner_name)), '\\s+', ' ', 'g'), '') AS normalized_name,
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(owner_address)), '\\s+', ' ', 'g'), '') AS normalized_address,
            vehicle_reg_no AS source_record_id,
            'EXCISE' AS source_database
        FROM read_csv('excise_vehicles.csv', all_varchar=true)

        UNION ALL

        -- Property Transactions
        SELECT
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(buyer_name)), '\\s+', ' ', 'g'), '') AS normalized_name,
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(property_address)), '\\s+', ' ', 'g'), '') AS normalized_address,
            registry_deed_no AS source_record_id,
            'PROPERTY' AS source_database
        FROM read_csv('property_transactions.csv', all_varchar=true)

        UNION ALL

        -- Utility Bills (DISCO)
        SELECT
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(consumer_name)), '\\s+', ' ', 'g'), '') AS normalized_name,
            COALESCE(REGEXP_REPLACE(TRIM(LOWER(installation_address)), '\\s+', ' ', 'g'), '') AS normalized_address,
            meter_ref_no AS source_record_id,
            'DISCO' AS source_database
        FROM read_csv('utility_bills.csv', all_varchar=true)

    ) TO 'layer1_ingested_master.parquet' (FORMAT PARQUET, CODEC 'SNAPPY');
    """

    print("Executing massive UNION ALL query for normalization, sanitization, and parquet export...")
    con.execute(query)
    
    elapsed = time.time() - start_time
    print(f"Data ingestion completed successfully in {elapsed:.2f} seconds.")
    print("Output saved to 'layer1_ingested_master.parquet'.")

if __name__ == "__main__":
    build_ingestion_layer()
