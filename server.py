import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ──────────────────────────────────────────────────────────────────────────────
# § 1. ENVIRONMENT & PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "layer5_audit_manifest.json")

def load_manifest_data():
    """Robustly parse the layer5 JSON manifest file."""
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Audit manifest not found at {MANIFEST_PATH}. Please run Layer 5 first.")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON manifest: {e}")

# Try to load data at startup
try:
    AUDIT_DATA = load_manifest_data()
except Exception as e:
    print(f"Startup Warning: {e}")
    AUDIT_DATA = []

# Initialize FastAPI App
app = FastAPI(
    title="GovTech Layer 5 Audit API",
    description="Serves Layer 5 ExAI audit trails to the Next.js frontend.",
    version="1.0.0"
)

# ──────────────────────────────────────────────────────────────────────────────
# § 2. CORS SECURITY CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# § 3. PYDANTIC DATA SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────
class ProfileSchema(BaseModel):
    canonical_id: str
    total_declared_income: float
    total_visible_wealth_pkr: float
    annual_utility_bill_pkr: float
    gnn_structural_anomaly_score: float
    final_hybrid_risk_score: float
    audit_justification_notice: str
    audit_status: Optional[str] = "Pending Review"

class SystemMetricsSchema(BaseModel):
    total_high_risk_targets: int
    maximum_hybrid_risk_score: float
    aggregate_unexplained_wealth_pkr: float

class StatusUpdateRequest(BaseModel):
    status: str

# ──────────────────────────────────────────────────────────────────────────────
# § 4. API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/metrics", response_model=SystemMetricsSchema)
def get_metrics():
    """
    Dynamically compute system-wide metrics from the loaded JSON data.
    Returns total profiles, highest risk score, and aggregate unexplained wealth.
    """
    if not AUDIT_DATA:
        return SystemMetricsSchema(
            total_high_risk_targets=0,
            maximum_hybrid_risk_score=0.0,
            aggregate_unexplained_wealth_pkr=0.0
        )

    total_targets = len(AUDIT_DATA)
    max_score = max(item.get("final_hybrid_risk_score", 0.0) for item in AUDIT_DATA)
    
    # Unexplained wealth = Total Visible Wealth - Total Declared Income
    agg_wealth = sum(
        max(0, item.get("total_visible_wealth_pkr", 0.0) - item.get("total_declared_income", 0.0))
        for item in AUDIT_DATA
    )

    return SystemMetricsSchema(
        total_high_risk_targets=total_targets,
        maximum_hybrid_risk_score=max_score,
        aggregate_unexplained_wealth_pkr=agg_wealth
    )

@app.get("/api/profiles", response_model=List[ProfileSchema])
def get_profiles():
    """
    Return the complete list of profiles, explicitly sorted in descending
    order by final_hybrid_risk_score.
    """
    # Create the schema instances (adds the default audit_status if missing)
    profiles = [ProfileSchema(**item) for item in AUDIT_DATA]
    # Sort descending
    profiles.sort(key=lambda x: x.final_hybrid_risk_score, reverse=True)
    return profiles

@app.get("/api/profiles/{canonical_id}", response_model=ProfileSchema)
def get_profile(canonical_id: str):
    """
    Look up and return a single individual's full schema by canonical_id.
    Raises 404 if not found.
    """
    for item in AUDIT_DATA:
        if item.get("canonical_id") == canonical_id:
            return ProfileSchema(**item)
    
    raise HTTPException(status_code=404, detail=f"Profile with ID {canonical_id} not found.")

@app.post("/api/profiles/{canonical_id}/status", response_model=ProfileSchema)
def update_profile_status(canonical_id: str, request: StatusUpdateRequest):
    """
    Simulate updating an individual's investigation state.
    Updates the in-memory data and persists it back to disk.
    """
    for item in AUDIT_DATA:
        if item.get("canonical_id") == canonical_id:
            item["audit_status"] = request.status
            # Persist to disk
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(AUDIT_DATA, f, indent=4)
                
            return ProfileSchema(**item)
            
    raise HTTPException(status_code=404, detail=f"Profile with ID {canonical_id} not found.")

# ──────────────────────────────────────────────────────────────────────────────
# § 5. SERVER EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting FastAPI Server...")
    # Run with uvicorn programmatically, hot-reloading active
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
