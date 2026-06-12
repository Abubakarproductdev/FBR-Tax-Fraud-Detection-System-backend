"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3: Cross-Asset Aggregation & Risk Profiling Engine                  ║
║                                                                            ║
║  Joins resolved entity mappings (Layer 2) to all source datasets,          ║
║  builds per-canonical_id financial dossiers via DuckDB, then applies       ║
║  a configurable OOP risk engine to produce evasion deviation scores.       ║
║                                                                            ║
║  Author : Senior Data Engineering Pipeline                                 ║
║  Inputs : layer2_resolved_entities.csv, fbr_returns.csv,                   ║
║           excise_vehicles.csv, property_transactions.csv,                  ║
║           utility_bills.csv                                                ║
║  Output : layer3_financial_profiles.csv                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import math
import duckdb
import pandas as pd
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# § 1.  VEHICLE MARKET VALUE ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────
# The excise_vehicles.csv has no `market_value` column.  We derive an
# estimated market value from (vehicle_make_model, engine_capacity_cc,
# registration_year) using a lookup of base prices per make-model and a
# simple depreciation curve.

VEHICLE_BASE_PRICES_PKR = {
    "Suzuki Mehran":          1_200_000,
    "Suzuki Alto":            2_400_000,
    "Suzuki Cultus":          2_800_000,
    "Honda City":             4_500_000,
    "Toyota Yaris":           5_000_000,
    "Suzuki Swift":           4_200_000,
    "Toyota Corolla":         6_500_000,
    "Honda Civic":            7_500_000,
    "Toyota Fortuner":       12_000_000,
    "Toyota Hilux Revo":     10_000_000,
    "Toyota Land Cruiser":   25_000_000,
    "Mercedes Benz S-Class": 40_000_000,
    "Audi E-Tron":           18_000_000,
}

CURRENT_YEAR = 2026


def estimate_vehicle_market_value(make_model: str, engine_cc: int, reg_year: int) -> float:
    """
    Estimate market value using a base price lookup + 7% annual depreciation,
    with a floor of 15% of the base price.  Unknown models get a heuristic
    based purely on engine displacement.
    """
    base = VEHICLE_BASE_PRICES_PKR.get(make_model)
    if base is None:
        # Fallback: PKR 3,000 per CC (rough heuristic)
        base = engine_cc * 3_000

    age = max(CURRENT_YEAR - reg_year, 0)
    depreciation_factor = max(0.15, (1 - 0.07) ** age)   # 7% YoY, floor 15%
    return round(base * depreciation_factor)


# ──────────────────────────────────────────────────────────────────────────────
# § 2.  CONFIGURABLE RISK ENGINE  (Object-Oriented)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvasionRiskEngine:
    """
    Configurable risk scoring engine.

    Weights:
        vehicle_upkeep_multiplier  – fraction of vehicle portfolio assumed as
                                     annual running cost (fuel, insurance, etc.)
        property_upkeep_multiplier – fraction of property portfolio assumed as
                                     annual upkeep (tax, maintenance, etc.)
        utility_weight             – multiplier on annualized utility spend
        lifestyle_buffer_pkr       – minimum baseline lifestyle cost assumed
                                     for every citizen (food, clothing, etc.)
    """
    vehicle_upkeep_multiplier: float = 0.05
    property_upkeep_multiplier: float = 0.02
    utility_weight: float = 1.0
    lifestyle_buffer_pkr: float = 300_000.0   # ~25k/month minimum

    def calculate_risk(self, row: pd.Series) -> pd.Series:
        """
        For a single dossier row, compute:
          • estimated_annual_lifestyle_cost
          • wealth_gap
          • deviation_score (0–100)
        Returns a Series with those three fields.
        """
        annual_utility = float(row.get("annual_utility_bill_pkr", 0))
        vehicle_foot   = float(row.get("vehicle_footprint_pkr", 0))
        property_foot  = float(row.get("property_footprint_pkr", 0))
        declared_income = float(row.get("total_declared_income", 0))

        # ── Estimated annual lifestyle cost ──
        estimated_cost = (
            (annual_utility * self.utility_weight)
            + (vehicle_foot * self.vehicle_upkeep_multiplier)
            + (property_foot * self.property_upkeep_multiplier)
            + self.lifestyle_buffer_pkr
        )

        # ── Wealth gap ──
        wealth_gap = estimated_cost - declared_income

        # ── Deviation score (0–100) ──
        #   If gap ≤ 0 → score = 0  (income covers lifestyle)
        #   Otherwise, scale relative to income (or estimated cost if income = 0)
        if wealth_gap <= 0:
            score = 0.0
        else:
            denominator = max(declared_income, estimated_cost, 1.0)
            raw = (wealth_gap / denominator) * 100
            score = min(raw, 100.0)

        return pd.Series({
            "estimated_annual_lifestyle_cost": round(estimated_cost, 2),
            "wealth_gap": round(wealth_gap, 2),
            "deviation_score": round(score, 2),
        })


# ──────────────────────────────────────────────────────────────────────────────
# § 3.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── File paths ──
    layer2_path   = os.path.join(BASE_DIR, "layer2_resolved_entities.csv")
    fbr_path      = os.path.join(BASE_DIR, "fbr_returns.csv")
    excise_path   = os.path.join(BASE_DIR, "excise_vehicles.csv")
    property_path = os.path.join(BASE_DIR, "property_transactions.csv")
    utility_path  = os.path.join(BASE_DIR, "utility_bills.csv")
    output_path   = os.path.join(BASE_DIR, "layer3_financial_profiles.csv")

    print("=" * 78)
    print("  LAYER 3 ── Cross-Asset Aggregation & Risk Profiling Engine")
    print("=" * 78)

    # ──────────────────────────────────────────────────────────────────────
    # 3-A.  Pre-process vehicles: add estimated_market_value_pkr column
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading & enriching excise_vehicles with market value estimates...")
    vehicles_df = pd.read_csv(excise_path, dtype=str)
    vehicles_df["engine_capacity_cc"] = pd.to_numeric(vehicles_df["engine_capacity_cc"], errors="coerce").fillna(0).astype(int)
    vehicles_df["registration_year"]  = pd.to_numeric(vehicles_df["registration_year"], errors="coerce").fillna(CURRENT_YEAR).astype(int)
    vehicles_df["estimated_market_value_pkr"] = vehicles_df.apply(
        lambda r: estimate_vehicle_market_value(
            r["vehicle_make_model"],
            r["engine_capacity_cc"],
            r["registration_year"],
        ),
        axis=1,
    )
    print(f"       ✓ {len(vehicles_df):,} vehicle records enriched.")

    # ──────────────────────────────────────────────────────────────────────
    # 3-B.  DuckDB: In-memory aggregation
    # ──────────────────────────────────────────────────────────────────────
    print("\n[2/5] Initialising DuckDB in-memory engine & loading datasets...")
    con = duckdb.connect(database=":memory:")

    # Register CSVs as virtual tables (auto-detect types)
    con.execute(f"CREATE TABLE layer2 AS SELECT * FROM read_csv_auto('{layer2_path.replace(chr(92), '/')}')")
    con.execute(f"CREATE TABLE fbr    AS SELECT * FROM read_csv_auto('{fbr_path.replace(chr(92), '/')}')")
    con.execute(f"CREATE TABLE property AS SELECT * FROM read_csv_auto('{property_path.replace(chr(92), '/')}')")
    con.execute(f"CREATE TABLE utility  AS SELECT * FROM read_csv_auto('{utility_path.replace(chr(92), '/')}')")

    # Register the enriched vehicles DataFrame directly
    con.register("vehicles_df", vehicles_df)
    con.execute("CREATE TABLE excise AS SELECT * FROM vehicles_df")

    # Quick sanity counts
    for tbl in ["layer2", "fbr", "excise", "property", "utility"]:
        cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"       ✓ {tbl:>10s}  →  {cnt:>6,} rows")

    # ──────────────────────────────────────────────────────────────────────
    # 3-C.  Build the Dossier table via SQL
    # ──────────────────────────────────────────────────────────────────────
    print("\n[3/5] Running cross-asset aggregation SQL...")

    DOSSIER_SQL = """
    WITH
    -- Map FBR records to canonical IDs and aggregate
    fbr_agg AS (
        SELECT
            l2.canonical_id,
            COALESCE(SUM(CAST(f.declared_income_pkr AS DOUBLE)), 0) AS total_declared_income,
            COALESCE(SUM(CAST(f.tax_paid_pkr        AS DOUBLE)), 0) AS total_tax_paid
        FROM layer2 l2
        INNER JOIN fbr f ON l2.source_record_id = f.fbr_id
        WHERE UPPER(l2.source_database) = 'FBR'
        GROUP BY l2.canonical_id
    ),

    -- Map Excise records to canonical IDs and aggregate
    vehicle_agg AS (
        SELECT
            l2.canonical_id,
            COALESCE(SUM(CAST(e.estimated_market_value_pkr AS DOUBLE)), 0) AS vehicle_footprint_pkr
        FROM layer2 l2
        INNER JOIN excise e ON l2.source_record_id = e.vehicle_reg_no
        WHERE UPPER(l2.source_database) = 'EXCISE'
        GROUP BY l2.canonical_id
    ),

    -- Map Property records to canonical IDs and aggregate
    property_agg AS (
        SELECT
            l2.canonical_id,
            COALESCE(SUM(CAST(p.property_value_pkr AS DOUBLE)), 0) AS property_footprint_pkr
        FROM layer2 l2
        INNER JOIN property p ON l2.source_record_id = p.registry_deed_no
        WHERE UPPER(l2.source_database) = 'PROPERTY'
        GROUP BY l2.canonical_id
    ),

    -- Map Utility records to canonical IDs and aggregate (annualize)
    utility_agg AS (
        SELECT
            l2.canonical_id,
            COALESCE(SUM(CAST(u.avg_monthly_bill_pkr AS DOUBLE) * 12), 0) AS annual_utility_bill_pkr
        FROM layer2 l2
        INNER JOIN utility u ON l2.source_record_id = u.meter_ref_no
        WHERE UPPER(l2.source_database) = 'DISCO'
        GROUP BY l2.canonical_id
    ),

    -- Get distinct canonical IDs
    all_ids AS (
        SELECT DISTINCT canonical_id FROM layer2
    )

    -- Final dossier: LEFT JOIN all aggregations onto the full ID list
    SELECT
        a.canonical_id,
        COALESCE(f.total_declared_income,    0) AS total_declared_income,
        COALESCE(f.total_tax_paid,           0) AS total_tax_paid,
        COALESCE(v.vehicle_footprint_pkr,    0) AS vehicle_footprint_pkr,
        COALESCE(p.property_footprint_pkr,   0) AS property_footprint_pkr,
        COALESCE(u.annual_utility_bill_pkr,  0) AS annual_utility_bill_pkr
    FROM all_ids a
    LEFT JOIN fbr_agg      f ON a.canonical_id = f.canonical_id
    LEFT JOIN vehicle_agg  v ON a.canonical_id = v.canonical_id
    LEFT JOIN property_agg p ON a.canonical_id = p.canonical_id
    LEFT JOIN utility_agg  u ON a.canonical_id = u.canonical_id
    ORDER BY a.canonical_id
    """

    dossier_df = con.execute(DOSSIER_SQL).fetchdf()
    con.close()

    print(f"       ✓ Dossier built: {len(dossier_df):,} canonical entities.")
    print(f"       Columns: {list(dossier_df.columns)}")

    # ──────────────────────────────────────────────────────────────────────
    # 3-D.  Apply the Risk Engine
    # ──────────────────────────────────────────────────────────────────────
    print("\n[4/5] Applying EvasionRiskEngine (OOP, configurable weights)...")
    engine = EvasionRiskEngine(
        vehicle_upkeep_multiplier=0.05,
        property_upkeep_multiplier=0.02,
        utility_weight=1.0,
        lifestyle_buffer_pkr=300_000.0,
    )

    risk_cols = dossier_df.apply(engine.calculate_risk, axis=1)
    dossier_df = pd.concat([dossier_df, risk_cols], axis=1)

    # Sort by deviation_score descending
    dossier_df.sort_values("deviation_score", ascending=False, inplace=True)
    dossier_df.reset_index(drop=True, inplace=True)

    # ──────────────────────────────────────────────────────────────────────
    # 3-E.  Export & Summary
    # ──────────────────────────────────────────────────────────────────────
    print("\n[5/5] Exporting to layer3_financial_profiles.csv...")
    dossier_df.to_csv(output_path, index=False)
    print(f"       ✓ Saved → {output_path}")

    # ── Top-5 highest-risk canonical IDs ──
    print("\n" + "═" * 78)
    print("  TOP 5 HIGHEST-RISK CANONICAL IDs")
    print("═" * 78)

    top5 = dossier_df.head(5)
    for idx, row in top5.iterrows():
        print(
            f"\n  ┌─ #{idx + 1}  canonical_id: {row['canonical_id']}"
        )
        print(f"  │   Declared Income:       PKR {row['total_declared_income']:>15,.0f}")
        print(f"  │   Tax Paid:              PKR {row['total_tax_paid']:>15,.0f}")
        print(f"  │   Vehicle Footprint:     PKR {row['vehicle_footprint_pkr']:>15,.0f}")
        print(f"  │   Property Footprint:    PKR {row['property_footprint_pkr']:>15,.0f}")
        print(f"  │   Annual Utility Bills:  PKR {row['annual_utility_bill_pkr']:>15,.0f}")
        print(f"  │   Est. Lifestyle Cost:   PKR {row['estimated_annual_lifestyle_cost']:>15,.0f}")
        print(f"  │   Wealth Gap:            PKR {row['wealth_gap']:>15,.0f}")
        print(f"  └─  DEVIATION SCORE:       {row['deviation_score']:>8.2f} / 100")

    # ── Summary statistics ──
    print("\n" + "─" * 78)
    high_risk    = len(dossier_df[dossier_df["deviation_score"] >= 70])
    medium_risk  = len(dossier_df[(dossier_df["deviation_score"] >= 40) & (dossier_df["deviation_score"] < 70)])
    low_risk     = len(dossier_df[dossier_df["deviation_score"] < 40])
    print(f"  Summary:  {len(dossier_df):,} entities profiled")
    print(f"     🔴 High Risk   (≥70):  {high_risk:,}")
    print(f"     🟡 Medium Risk (40–69): {medium_risk:,}")
    print(f"     🟢 Low Risk    (<40):   {low_risk:,}")
    print("─" * 78)
    print("  Layer 3 pipeline complete.\n")


if __name__ == "__main__":
    main()
