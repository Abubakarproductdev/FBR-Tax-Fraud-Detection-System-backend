"""
==========================================================================
 Layer 2: The Entity Resolution Engine
 GovTech Pipeline — Senior ML Engineering Implementation
==========================================================================
 Pipeline Stages:
   1. Semantic Storage   — Local SentenceTransformer Embeddings → Pinecone Vector DB
   2. LSH Blocking       — MinHash Locality-Sensitive Hashing for Candidate Generation
   3. Pairwise Scoring   — Jaro-Winkler (Name) + Cosine Similarity (Address) → NetworkX Clustering
==========================================================================
"""

import os
import time
import uuid
import hashlib
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import networkx as nx
from dotenv import load_dotenv
from datasketch import MinHash, MinHashLSH
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import token_set_ratio
from pinecone import Pinecone, ServerlessSpec

warnings.filterwarnings("ignore")

# =====================================================================
# Configuration Constants
# =====================================================================
MODEL_NAME = "intfloat/multilingual-e5-base"
PINECONE_INDEX_NAME = "govtech-addresses"
EMBEDDING_DIM = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# LSH Parameters
LSH_THRESHOLD = 0.5
LSH_NUM_PERM = 128

# Scoring Thresholds
NAME_SCORE_THRESHOLD = 0.80
ADDRESS_SCORE_THRESHOLD = 0.85

# Batching Parameters
EMBED_BATCH_SIZE = 64       # Texts per local encode batch
PINECONE_BATCH_SIZE = 100   # Vectors per Pinecone upsert


# =====================================================================
# STAGE 0: Environment & Data Loading
# =====================================================================
def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    pinecone_api_key = os.getenv("PINECORN_API_KEY")

    if not pinecone_api_key:
        raise EnvironmentError("Missing 'PINECORN_API_KEY' in .env file.")

    print("[ENV] ✓ Environment variables loaded successfully.")
    return pinecone_api_key


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load the Layer 1 ingested master parquet file."""
    print(f"[DATA] Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[DATA] ✓ Loaded {len(df):,} records with columns: {df.columns.tolist()}")
    print(f"[DATA]   Unique names: {df['normalized_name'].nunique():,}")
    print(f"[DATA]   Unique addresses: {df['normalized_address'].nunique():,}")
    return df


# =====================================================================
# STAGE 1: Semantic Storage (HuggingFace Embeddings + Pinecone)
# =====================================================================
def init_pinecone(api_key: str):
    """Initialize Pinecone client and ensure the index exists."""
    print("[PINECONE] Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[PINECONE] Index '{PINECONE_INDEX_NAME}' not found. Creating...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for index to be ready
        print("[PINECONE] Waiting for index to become ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
            time.sleep(2)
        print(f"[PINECONE] ✓ Index '{PINECONE_INDEX_NAME}' created and ready.")
    else:
        print(f"[PINECONE] ✓ Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"[PINECONE] Index stats — Total vectors: {stats.total_vector_count:,}")
    return index


def address_to_id(address: str) -> str:
    """Generate a deterministic ID from an address string using SHA-256 hash."""
    return hashlib.sha256(address.encode("utf-8")).hexdigest()[:32]


def build_semantic_store(df: pd.DataFrame, pinecone_index) -> dict:
    """
    Stage 1: Embed all unique addresses locally via SentenceTransformer
    and upsert into Pinecone.
    Returns a local dictionary mapping address → embedding vector for Stage 3.
    """
    unique_addresses = df["normalized_address"].dropna().unique().tolist()
    unique_addresses = [a for a in unique_addresses if a.strip()]
    total = len(unique_addresses)
    print(f"\n{'='*70}")
    print(f" STAGE 1: Semantic Storage — {total:,} unique addresses")
    print(f"{'='*70}")

    # --- Load the model locally (downloads on first run, cached after) ---
    print(f"[MODEL] Loading SentenceTransformer('{MODEL_NAME}')...")
    model_start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[MODEL] ✓ Model loaded in {time.time() - model_start:.2f}s")

    address_vectors = {}  # address_text → numpy vector (local cache)
    all_upsert_records = []

    # Prepend "query: " prefix as required by E5 model family
    prefixed_addresses = [f"query: {a}" for a in unique_addresses]

    # --- Embed in batches ---
    total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    embed_start = time.time()

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, total)
        batch_texts = prefixed_addresses[batch_start:batch_end]
        batch_originals = unique_addresses[batch_start:batch_end]
        batch_num = (batch_start // EMBED_BATCH_SIZE) + 1

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for text, vec in zip(batch_originals, embeddings):
            vec_np = np.array(vec, dtype=np.float32)
            address_vectors[text] = vec_np
            all_upsert_records.append({
                "id": address_to_id(text),
                "values": vec_np.tolist(),
                "metadata": {"address_text": text},
            })

        print(f"[EMBED] Batch {batch_num}/{total_batches} "
              f"({batch_end}/{total} addresses encoded)")

    embed_elapsed = time.time() - embed_start
    print(f"[EMBED] ✓ Embedded {len(address_vectors):,} addresses "
          f"into {EMBEDDING_DIM}-dim vectors in {embed_elapsed:.2f}s")

    # --- Upsert into Pinecone in batches ---
    print(f"[PINECONE] Upserting {len(all_upsert_records):,} vectors...")
    for i in range(0, len(all_upsert_records), PINECONE_BATCH_SIZE):
        batch = all_upsert_records[i:i + PINECONE_BATCH_SIZE]
        pinecone_index.upsert(vectors=batch)
        upserted_so_far = min(i + PINECONE_BATCH_SIZE, len(all_upsert_records))
        print(f"[PINECONE] ↑ Upserted {upserted_so_far:,}/{len(all_upsert_records):,}")

    print(f"[PINECONE] ✓ All vectors upserted successfully.")
    return address_vectors


# =====================================================================
# STAGE 2: LSH Blocking (Candidate Generation)
# =====================================================================
def tokenize_name(name: str) -> set:
    """Tokenize a normalized name into character n-grams (3-grams) for MinHash."""
    name = str(name).strip().lower()
    if len(name) < 3:
        return set(name)
    return set(name[i:i+3] for i in range(len(name) - 2))


def build_lsh_index(df: pd.DataFrame) -> set:
    """
    Stage 2: Build MinHash LSH index on normalized_name and generate candidate pairs.
    Returns a set of (row_idx_i, row_idx_j) tuples where i < j.
    """
    print(f"\n{'='*70}")
    print(f" STAGE 2: LSH Blocking — Candidate Generation")
    print(f"{'='*70}")

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
    minhashes = {}

    print(f"[LSH] Building MinHash signatures for {len(df):,} records...")
    start = time.time()

    for idx, row in df.iterrows():
        tokens = tokenize_name(row["normalized_name"])
        m = MinHash(num_perm=LSH_NUM_PERM)
        for token in tokens:
            m.update(token.encode("utf-8"))
        minhashes[idx] = m

        try:
            lsh.insert(str(idx), m)
        except ValueError:
            # Duplicate key — skip
            pass

    elapsed = time.time() - start
    print(f"[LSH] ✓ MinHash signatures built in {elapsed:.2f}s")

    # --- Query for candidate pairs ---
    print("[LSH] Querying for candidate pairs...")
    candidate_pairs = set()

    for idx, m in minhashes.items():
        results = lsh.query(m)
        for other in results:
            other_idx = int(other)
            if idx != other_idx:
                pair = (min(idx, other_idx), max(idx, other_idx))
                candidate_pairs.add(pair)

    print(f"[LSH] ✓ Generated {len(candidate_pairs):,} candidate pairs")
    return candidate_pairs, minhashes


# =====================================================================
# STAGE 3: Pairwise Scoring & Clustering
# =====================================================================
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def score_and_cluster(
    df: pd.DataFrame,
    candidate_pairs: set,
    address_vectors: dict,
) -> pd.DataFrame:
    """
    Stage 3: Score candidate pairs and cluster matching entities.
    
    For each candidate pair:
      - Name Score:    JaroWinkler normalized similarity
      - Address Score: Cosine similarity of embedding vectors (fallback: token_set_ratio)
    
    If both thresholds are met, an edge is added to the resolution graph.
    Connected components form entity clusters, each assigned a canonical UUID.
    """
    print(f"\n{'='*70}")
    print(f" STAGE 3: Pairwise Scoring & Clustering")
    print(f"{'='*70}")

    G = nx.Graph()
    total_pairs = len(candidate_pairs)
    matched_edges = 0
    vector_hits = 0
    vector_misses = 0

    print(f"[SCORING] Evaluating {total_pairs:,} candidate pairs...")
    start = time.time()

    for pair_idx, (i, j) in enumerate(candidate_pairs, 1):
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        # --- Name Score (Jaro-Winkler) ---
        name_score = JaroWinkler.normalized_similarity(
            row_i["normalized_name"], row_j["normalized_name"]
        )

        if name_score <= NAME_SCORE_THRESHOLD:
            continue  # Early exit — no point computing address score

        # --- Address Score (Cosine Similarity with Fallback) ---
        addr_i = row_i["normalized_address"]
        addr_j = row_j["normalized_address"]
        vec_i = address_vectors.get(addr_i)
        vec_j = address_vectors.get(addr_j)

        if vec_i is not None and vec_j is not None:
            address_score = cosine_similarity(vec_i, vec_j)
            vector_hits += 1
        else:
            # Fallback: RapidFuzz token set ratio (scaled to 0-1)
            address_score = token_set_ratio(addr_i, addr_j) / 100.0
            vector_misses += 1

        # --- Decision: Add edge if both thresholds met ---
        if name_score > NAME_SCORE_THRESHOLD and address_score > ADDRESS_SCORE_THRESHOLD:
            src_id_i = row_i["source_record_id"]
            src_id_j = row_j["source_record_id"]
            G.add_edge(src_id_i, src_id_j, name_score=name_score, address_score=address_score)
            matched_edges += 1

        # Progress logging
        if pair_idx % 5000 == 0 or pair_idx == total_pairs:
            print(f"[SCORING] Processed {pair_idx:,}/{total_pairs:,} pairs | "
                  f"Edges: {matched_edges:,} | "
                  f"Vector hits: {vector_hits:,} / misses: {vector_misses:,}")

    elapsed = time.time() - start
    print(f"[SCORING] ✓ Scoring complete in {elapsed:.2f}s")
    print(f"[SCORING]   Total edges added: {matched_edges:,}")
    print(f"[SCORING]   Graph nodes: {G.number_of_nodes():,}")

    # --- Extract connected components and assign canonical IDs ---
    print("\n[CLUSTER] Extracting connected components...")
    components = list(nx.connected_components(G))
    print(f"[CLUSTER] ✓ Found {len(components):,} multi-record clusters")

    # Build mapping: source_record_id → canonical_id
    record_to_canonical = {}
    cluster_sizes = []

    for component in components:
        canonical_id = str(uuid.uuid4())
        cluster_sizes.append(len(component))
        for record_id in component:
            record_to_canonical[record_id] = canonical_id

    if cluster_sizes:
        print(f"[CLUSTER]   Largest cluster:  {max(cluster_sizes)} records")
        print(f"[CLUSTER]   Average cluster:  {np.mean(cluster_sizes):.1f} records")
        print(f"[CLUSTER]   Median cluster:   {np.median(cluster_sizes):.0f} records")

    # --- Build final output DataFrame ---
    print("\n[OUTPUT] Building resolved entity dataframe...")

    # Assign canonical IDs — singletons get their own UUID
    df["canonical_id"] = df["source_record_id"].apply(
        lambda rid: record_to_canonical.get(rid, str(uuid.uuid4()))
    )

    output_df = df[["canonical_id", "source_database", "source_record_id",
                     "normalized_name", "normalized_address"]].copy()

    return output_df


# =====================================================================
# MAIN PIPELINE ORCHESTRATOR
# =====================================================================
def run_entity_resolution_pipeline():
    """Execute the full Layer 2 Entity Resolution Engine."""
    pipeline_start = time.time()

    print("=" * 70)
    print(" 🔍  LAYER 2: THE ENTITY RESOLUTION ENGINE")
    print(" 🏛️  GovTech Pipeline — Entity Deduplication & Linking")
    print("=" * 70)

    # --- Stage 0: Setup ---
    pinecone_api_key = load_environment()
    df = load_data("layer1_ingested_master.parquet")

    # --- Stage 1: Semantic Storage ---
    pinecone_index = init_pinecone(pinecone_api_key)
    address_vectors = build_semantic_store(df, pinecone_index)

    # --- Stage 2: LSH Blocking ---
    candidate_pairs, _ = build_lsh_index(df)

    # --- Stage 3: Pairwise Scoring & Clustering ---
    output_df = score_and_cluster(df, candidate_pairs, address_vectors)

    # --- Export ---
    output_path = "layer2_resolved_entities.csv"
    output_df.to_csv(output_path, index=False)

    total_entities = output_df["canonical_id"].nunique()
    pipeline_elapsed = time.time() - pipeline_start

    print(f"\n{'='*70}")
    print(f" ✅  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"[RESULT] Total input records:     {len(output_df):,}")
    print(f"[RESULT] Resolved entities:       {total_entities:,}")
    print(f"[RESULT] Dedup ratio:             {(1 - total_entities/len(output_df))*100:.1f}%")
    print(f"[RESULT] Output saved to:         {output_path}")
    print(f"[RESULT] Pipeline runtime:        {pipeline_elapsed:.2f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_entity_resolution_pipeline()
