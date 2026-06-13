import os
import json
import subprocess
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys

# ──────────────────────────────────────────────────────────────────────────────
# § 1. ENVIRONMENT & PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "layer5_audit_manifest.json")

# Initialize FastAPI App
app = FastAPI(
    title="GovTech AI Orchestrator API",
    description="Dynamic Pipeline Orchestrator to ingest data, run AI models, and serve audit trails.",
    version="2.0.0"
)

# ──────────────────────────────────────────────────────────────────────────────
# § 2. CORS SECURITY CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
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
    gds_structural_anomaly_score: float  # Updated to match Memgraph architecture
    final_hybrid_risk_score: float
    audit_justification_notice: str
    audit_status: Optional[str] = "Pending Review"

class PipelineResponse(BaseModel):
    status: str
    message: str

class SystemMetrics(BaseModel):
    total_high_risk_targets: int
    maximum_hybrid_risk_score: float
    aggregate_unexplained_wealth_pkr: float

class StatusUpdateRequest(BaseModel):
    status: str


def read_manifest() -> List[dict]:
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Audit manifest not found. Has the pipeline been run?")

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Corrupted manifest JSON: {str(e)}")

    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Corrupted manifest JSON: expected a list of profiles.")

    return data


def write_manifest(data: List[dict]) -> None:
    try:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist manifest: {str(e)}")


def validate_profiles(data: List[dict]) -> List[ProfileSchema]:
    try:
        profiles = [ProfileSchema(**item) for item in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate profiles: {str(e)}")

    profiles.sort(key=lambda x: x.final_hybrid_risk_score, reverse=True)
    return profiles

# ──────────────────────────────────────────────────────────────────────────────
# § 4. API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accept multiple CSV files from the frontend and save them directly 
    to the root project directory, overwriting existing files.
    """
    saved_files = []
    
    try:
        for file in files:
            safe_filename = os.path.basename(file.filename)
            if not safe_filename:
                raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

            file_path = os.path.join(BASE_DIR, safe_filename)
            content = await file.read()
            # Overwrite any existing files
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(safe_filename)
            
        return {"status": "success", "saved_files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/api/run-pipeline", response_model=PipelineResponse)
def run_pipeline():
    """
    Execute the entire GovTech AI pipeline sequentially via subprocesses.
    Halts execution and returns 500 if any layer fails.
    """
    pipeline_scripts = [
        "defensive_ingestion.py",
        "entity_resolution_engine.py",
        "layer3_aggregation_engine.py",
        "layer3a_kg_materialization.py",
        "layer4_hybrid_anomaly_engine.py",
        "layer5_explainable_ai.py"
    ]
    
    python_exe = sys.executable
    
    # Explicitly pass the current environment variables so Groq/Memgraph keys are inherited
    env = os.environ.copy()

    for script in pipeline_scripts:
        script_path = os.path.join(BASE_DIR, script)
        
        if not os.path.exists(script_path):
            raise HTTPException(status_code=500, detail=f"Pipeline Error: Script {script} not found.")
            
        try:
            print(f"Executing: {script} ...")
            # Execute synchronously. check=True throws CalledProcessError on non-zero exit code.
            result = subprocess.run(
                [python_exe, script_path], 
                cwd=BASE_DIR, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            print(f"Finished {script} successfully.")
            
        except subprocess.CalledProcessError as e:
            # Provide exact details of the script failure to the frontend
            error_message = f"Pipeline failed at {script}. Exit Code: {e.returncode}\nError Log: {e.stderr}"
            print(error_message)
            raise HTTPException(status_code=500, detail=error_message)

    return PipelineResponse(status="success", message="Pipeline executed completely.")


@app.get("/api/profiles", response_model=List[ProfileSchema])
def get_profiles():
    """
    Read the dynamically generated layer5_audit_manifest.json and return it.
    Validates output using ProfileSchema and explicitly sorts by risk score.
    """
    return validate_profiles(read_manifest())


@app.get("/api/metrics", response_model=SystemMetrics)
def get_metrics():
    profiles = validate_profiles(read_manifest())
    high_risk_profiles = [profile for profile in profiles if profile.final_hybrid_risk_score >= 80]

    return SystemMetrics(
        total_high_risk_targets=len(high_risk_profiles),
        maximum_hybrid_risk_score=max((profile.final_hybrid_risk_score for profile in profiles), default=0.0),
        aggregate_unexplained_wealth_pkr=sum(
            max(
                0.0,
                profile.total_visible_wealth_pkr
                + profile.annual_utility_bill_pkr
                - profile.total_declared_income,
            )
            for profile in high_risk_profiles
        ),
    )


@app.post("/api/profiles/{canonical_id}/status", response_model=PipelineResponse)
def update_profile_status(canonical_id: str, payload: StatusUpdateRequest):
    status = payload.status.strip()
    if not status:
        raise HTTPException(status_code=400, detail="Status is required.")

    data = read_manifest()
    for item in data:
        if item.get("canonical_id") == canonical_id:
            item["audit_status"] = status
            write_manifest(data)
            return PipelineResponse(status="success", message="Audit status updated.")

    raise HTTPException(status_code=404, detail=f"Profile {canonical_id} not found.")

# ──────────────────────────────────────────────────────────────────────────────
# § 5. SERVER EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting FastAPI Dynamic Orchestrator Server...")
    # Run with uvicorn programmatically, hot-reloading active
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)