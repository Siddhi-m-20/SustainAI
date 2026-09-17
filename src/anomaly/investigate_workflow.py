import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_data import validate_multi_resource_data
from src.analytics.kpi_engine import calculate_unified_campus_kpis
from src.anomaly.anomaly_service import AnomalyService


def run_investigation_pipeline(
    file_path: Path,
    min_severity: float = 40.0,
    filter_resource: str = "ALL",
    filter_building: str = "ALL",
    filter_tier: str = "ALL",
    inspect_top_n: int = 5,
):
    print("=" * 80)
    print(" ECOSYNC RESOURCE MANAGER — PHASE 1 INVESTIGATE WORKFLOW")
    print("=" * 80)
    print(f"Loading and validating dataset: {file_path}")

    # 1. Validation & Ingestion
    val_result = validate_multi_resource_data(file_path)
    if not val_result["is_valid"]:
        print("\n[ERROR] Validation failed:")
        for err in val_result["errors"]:
            print(f"  - {err}")
        return None, None

    if val_result["warnings"]:
        print("\n[WARNINGS] Dataset notices:")
        for warn in val_result["warnings"]:
            print(f"  - {warn}")

    df = val_result["cleaned_df"]
    schema = val_result["detected_schema"]
    print(f"\n[VALIDATION OK] Detected Schema: {schema.upper()}")
    print(f"Total records: {val_result['row_count']:,} | Facilities: {len(val_result['buildings'])} ({', '.join(val_result['buildings'])})")
    print(f"Time Range: {val_result['date_range'][0]} to {val_result['date_range'][1]}")

    # 2. Sustainability KPIs
    print("\n" + "-" * 80)
    print(" MULTI-RESOURCE SUSTAINABILITY SCORECARD (SDG 7, 6, 12)")
    print("-" * 80)
    kpis = calculate_unified_campus_kpis(df)

    if "total_kwh" in kpis["energy"]:
        e = kpis["energy"]
        print(f"  [SDG 7 Energy] Total: {e['total_kwh']:,.1f} kWh ({e['total_mwh']:.2f} MWh) | Peak: {e['peak_kw']:.1f} kW | Cost: ${e['estimated_cost_usd']:,.2f} | Carbon: {e['carbon_emissions_tco2e']:.2f} tCO2e")
    if "total_m3" in kpis["water"]:
        w = kpis["water"]
        print(f"  [SDG 6 Water]  Total: {w['total_m3']:,.1f} m3 ({w['total_liters']:,.0f} L) | Night-Flow Min: {w['min_night_flow_m3_h']:.2f} m3/h | Cost: ${w['estimated_cost_usd']:,.2f}")
    if "total_waste_kg" in kpis["waste"]:
        r = kpis["waste"]
        print(f"  [SDG 12 Waste] Total: {r['total_waste_kg']:,.1f} kg | Diversion Rate: {r['overall_diversion_rate_pct']:.1f}% | Landfill Carbon: {r['landfill_emissions_tco2e']:.2f} tCO2e")

    print(f"  --> Combined Campus Operational Footprint: ${kpis['total_operational_cost_usd']:,.2f} | {kpis['total_carbon_footprint_tco2e']:.2f} tCO2e")

    # 3. Anomaly Detection Execution
    print("\n" + "-" * 80)
    print(" EXECUTING ENSEMBLE ANOMALY DETECTION (Baseline + Rolling Z + Isolation Forest)")
    print("-" * 80)
    service = AnomalyService(contamination=0.03)
    scored_df, ledger = service.detect_multi_resource(df)

    if ledger.empty:
        print("[INFO] No anomalies exceeded the severity threshold.")
        return scored_df, ledger

    print(f"Total Anomaly Incidents Identified: {len(ledger):,}")

    # Breakdown by resource & tier
    summary_tier = ledger.groupby(["resource", "severity_tier"]).size().unstack(fill_value=0)
    print("\nIncident Summary by Resource and Severity Tier:")
    print(summary_tier.to_string())

    # 4. Filtering
    filtered = ledger.copy()
    if filter_resource != "ALL":
        filtered = filtered[filtered["resource"].str.upper() == filter_resource.upper()]
    if filter_building != "ALL":
        filtered = filtered[filtered["building"].str.upper() == filter_building.upper()]
    if filter_tier != "ALL":
        filtered = filtered[filtered["severity_tier"].str.upper() == filter_tier.upper()]

    # 5. Diagnostic Incident Inspection
    print("\n" + "-" * 80)
    print(f" TOP {min(inspect_top_n, len(filtered))} ANOMALY INCIDENTS (Filter: Resource={filter_resource}, Building={filter_building}, Tier={filter_tier})")
    print("-" * 80)

    top_incidents = filtered.head(inspect_top_n)
    for i, (_, inc) in enumerate(top_incidents.iterrows(), start=1):
        print(f"\n[{i}] Incident {inc['incident_id']} | Severity: {inc['severity_score']}/100 [{inc['severity_tier'].upper()}]")
        print(f"    Timestamp: {inc['timestamp']} | Location: {inc['building']} | Resource: {inc['resource']}")
        print(f"    Pattern:   {inc['pattern']}")
        print(f"    Readings:  Actual: {inc['actual_value']} {inc['unit']} | Expected Baseline: {inc['expected_value']} {inc['unit']} (Delta: +{inc['excess_consumption']} {inc['unit']})")
        print(f"    Deviation: {inc['deviation_percent']:+.1f}% from historical contextual median")
        print(f"    Impact:    Est. Financial Loss: ${inc['estimated_cost_loss_usd']:.2f} | Excess Carbon: {inc['excess_carbon_kg']:.2f} kg CO2e")

    print("\n" + "=" * 80)
    print(" PHASE 1 FOUNDATION VERIFIED: Ready for RAG & Agentic Investigation (Phase 2+)")
    print("=" * 80)
    return scored_df, ledger


def main():
    parser = argparse.ArgumentParser(description="EcoSync Phase 1 Multi-Resource Investigate Workflow")
    parser.add_argument("--file", type=str, default=None, help="Path to resource CSV file")
    parser.add_argument("--resource", type=str, default="ALL", help="Filter by resource (Energy, Water, Waste, ALL)")
    parser.add_argument("--building", type=str, default="ALL", help="Filter by building")
    parser.add_argument("--tier", type=str, default="ALL", help="Filter by severity tier (Critical, High, Medium, Low, ALL)")
    parser.add_argument("--top", type=int, default=5, help="Number of top incidents to inspect in detail")

    args = parser.parse_args()

    # Determine input file
    if args.file:
        data_path = Path(args.file)
    else:
        # Default to multi-resource with anomalies
        default_multi = PROJECT_ROOT / "data" / "processed" / "campus_multi_resource_with_anomalies.csv"
        if default_multi.exists():
            data_path = default_multi
        else:
            data_path = PROJECT_ROOT / "data" / "processed" / "campus_energy_with_anomalies.csv"

    run_investigation_pipeline(
        file_path=data_path,
        filter_resource=args.resource,
        filter_building=args.building,
        filter_tier=args.tier,
        inspect_top_n=args.top,
    )


if __name__ == "__main__":
    main()
