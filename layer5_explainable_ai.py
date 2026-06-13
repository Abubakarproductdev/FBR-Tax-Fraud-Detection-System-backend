"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 5: Explainable AI & Automated Audit Trails                            ║
║                                                                              ║
║  Extracts ALL individuals from the Layer 4 Hybrid Profiles and leverages     ║
║  the Groq API (llama3-8b-8192) to generate strictly formatted, legally-      ║
║  toned Audit Justification Notices.                                          ║
║  Outputs results to a JSON manifest for frontend consumption.                ║
║                                                                              ║
║  Author : Lead AI Engineer                                                   ║
║  Inputs : layer4_hybrid_profiles.csv, GROQ_API_KEY                           ║
║  Output : layer5_audit_manifest.json                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import textwrap
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# ──────────────────────────────────────────────────────────────────────────────
# § 1. SETUP & DATA EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4_PATH = os.path.join(BASE_DIR, "layer4_hybrid_profiles.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "layer5_audit_manifest.json")

def main():
    print("="*78)
    print("  LAYER 5 ── Explainable AI & Automated Audit Trails")
    print("="*78)

    print("\n[1/3] Loading Layer 4 Profiles...")
    try:
        df = pd.read_csv(L4_PATH)
    except FileNotFoundError:
        print(f"Error: {L4_PATH} not found. Please run Layer 4 first.")
        return

    # Extract ALL profiles, sorted by highest risk first
    target_profiles = df.sort_values(by="final_hybrid_risk_score", ascending=False)
    total_profiles = len(target_profiles)
    print(f"       ✓ Extracted all {total_profiles} profiles for full dashboard population.")

    # ──────────────────────────────────────────────────────────────────────────────
    # § 2. AGENTIC AI PROMPTING
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Generating AI Audit Justification Notices (via Groq llama-3.1-8b-instant)...")
    print("      (This may take 60-90 seconds due to API rate limits. Please wait.)\n")
    
    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = "You are a Senior FBR Forensic Auditor. Follow generation rules with absolute strictness."
    
    manifest_data = []
    current_count = 1

    for idx, row in target_profiles.iterrows():
        cid = row['canonical_id']
        income = float(row['total_declared_income'])
        # Wealth is the sum of property and vehicles
        wealth = float(row.get('property_footprint_pkr', 0) + row.get('vehicle_footprint_pkr', 0))
        utility = float(row.get('annual_utility_bill_pkr', 0))
        gds_score = float(row['gds_structural_anomaly_score'])
        hybrid_score = float(row['final_hybrid_risk_score'])

        user_prompt = f"""
        Profile Data:
        - Canonical ID: {cid}
        - Total Declared Income: PKR {income:,.0f}
        - Total Visible Wealth (Properties + Vehicles): PKR {wealth:,.0f}
        - Annual Utility Bills: PKR {utility:,.0f}
        - GDS Structural Anomaly Score: {gds_score:.2f} / 100

        Generation Rules:
        Generate an 'Audit Justification Notice' that is EXACTLY three sentences long. Do not include any greetings, bullet points, or extra text.
        Sentence 1: State the exact financial discrepancy between their declared income and their visible asset footprint or utility spend.
        Sentence 2: Highlight that our Graph Neural Network has flagged their asset network as highly anomalous and complex.
        Sentence 3: Formally initiate an asset verification audit explicitly citing Section 111 (Unexplained Income or Assets) of the Income Tax Ordinance, 2001.
        """

        try:
            # Call Groq API with Llama3
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temperature for highly deterministic, professional tone
                max_tokens=250
            )
            
            notice = response.choices[0].message.content.strip()

            print(f"  ┌─ PROFILE {current_count}/{total_profiles}: {cid}")
            print(f"  │  Hybrid Risk Score: {hybrid_score:.2f} / 100")
            print(f"  │  AI Audit Notice:")
            wrapped_notice = textwrap.fill(notice, width=70, initial_indent="  │    ", subsequent_indent="  │    ")
            print(f"{wrapped_notice}")
            print(f"  └─" + "─"*74)

            # Append to manifest list
            manifest_data.append({
                "canonical_id": cid,
                "total_declared_income": income,
                "total_visible_wealth_pkr": wealth,
                "annual_utility_bill_pkr": utility,
                "gds_structural_anomaly_score": gds_score,
                "final_hybrid_risk_score": hybrid_score,
                "audit_justification_notice": notice,
                "audit_status": "Pending Review" # Default status for the Next.js UI
            })
            
            # Brief sleep to avoid hitting rate limits on free/developer tiers
            time.sleep(1.5)

        except Exception as e:
            print(f"\n  [ERROR] Failed to generate notice for {cid}: {str(e)}")
            
        current_count += 1

    # ──────────────────────────────────────────────────────────────────────────────
    # § 3. OUTPUT & EXPORT
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Exporting to layer5_audit_manifest.json...")
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)
        print(f"       ✓ Saved JSON manifest -> {OUTPUT_PATH}")
    except Exception as e:
        print(f"       [ERROR] Failed to save JSON manifest: {str(e)}")
    
    print("\n" + "─"*78)
    print("  Layer 5 pipeline complete.\n")

if __name__ == "__main__":
    main()