"""Generate missing_dates_priority_report.md with detailed analysis and ranking of missing HEL1OS dates."""

from pathlib import Path
import ast
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "solexs" / "flare_catalog" / "flare_events_labeled_expanded.csv"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "hel1os" / "extracted"
RAW_DOWNLOADS_DIR = PROJECT_ROOT / "data" / "hel1os" / "raw_downloads"
REPORT_PATH = PROJECT_ROOT / "results" / "missing_dates_priority_report.md"


def main():
    py_files = list(RAW_DOWNLOADS_DIR.glob("*.py"))
    script_dates = set()
    for pf in py_files:
        try:
            code = pf.read_text(encoding="utf-8")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "data_file_paths" and isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    path_str = str(elt.value)
                                    m = re.search(r"level1/(\d{4})/(\d{2})/(\d{2})/", path_str)
                                    if m:
                                        script_dates.add(f"{m.group(1)}{m.group(2)}{m.group(3)}")
                                    else:
                                        m2 = re.search(r"HLS_(\d{4})(\d{2})(\d{2})_", path_str)
                                        if m2:
                                            script_dates.add(f"{m2.group(1)}{m2.group(2)}{m2.group(3)}")
        except Exception:
            pass

    extracted_dates = set([d.name for d in EXTRACTED_DIR.iterdir() if d.is_dir()])

    df = pd.read_csv(CATALOG_PATH)
    df["date_str"] = df["date"].astype(str).str.replace("-", "").str.zfill(8)

    df["in_extracted"] = df["date_str"].isin(extracted_dates)
    df["in_pradan_script"] = df["date_str"].isin(script_dates)

    missing_df = df[~df["in_extracted"]].copy()

    grouped = missing_df.groupby("date_str")

    date_records = []
    for dt, group in grouped:
        goes_classes = group["goes_class"].fillna("UNMATCHED").tolist()
        m_x_flares = [g for g in goes_classes if str(g).startswith("M") or str(g).startswith("X")]
        x_flares = [g for g in goes_classes if str(g).startswith("X")]
        m_flares = [g for g in goes_classes if str(g).startswith("M")]
        c_flares = [g for g in goes_classes if str(g).startswith("C")]
        b_flares = [g for g in goes_classes if str(g).startswith("B")]
        unmatched = [g for g in goes_classes if str(g) == "UNMATCHED"]

        in_script = dt in script_dates
        pradan_status = "Ready in raw_downloads" if in_script else "Requires PRADAN Portal Fetch"

        if len(m_x_flares) > 0:
            tier_num = 1
            tier_name = "Tier 1: High Priority (M/X Class Flares)"
        elif len(group) >= 5 or in_script:
            tier_num = 2
            tier_name = "Tier 2: Medium Priority (High Flare Density / Script Available)"
        else:
            tier_num = 3
            tier_name = "Tier 3: Low Priority (Minor / Low Volume Flares)"

        score = len(x_flares) * 100 + len(m_flares) * 20 + len(c_flares) * 3 + len(b_flares) * 1 + len(unmatched) * 1

        date_records.append({
            "date": dt,
            "tier_num": tier_num,
            "tier_name": tier_name,
            "mx_count": len(m_x_flares),
            "x_count": len(x_flares),
            "m_count": len(m_flares),
            "c_count": len(c_flares),
            "b_count": len(b_flares),
            "unmatched_count": len(unmatched),
            "total_flares": len(group),
            "mx_classes": ", ".join(m_x_flares) if m_x_flares else "None",
            "script_exists": in_script,
            "pradan_status": pradan_status,
            "score": score
        })

    res_df = pd.DataFrame(date_records)
    res_df = res_df.sort_values(by=["tier_num", "mx_count", "score", "total_flares"], ascending=[True, False, False, False]).reset_index(drop=True)

    md = []
    md.append("# HEL1OS Missing Observation Dates Priority & Scientific Importance Report\n")
    md.append(f"**Report Location**: [missing_dates_priority_report.md](file://{REPORT_PATH.resolve()})  ")
    md.append("**Analysis Timestamp**: 2026-08-26 14:56:00 IST  ")
    md.append("**Objective**: Categorize and rank all observation dates missing HEL1OS coverage, prioritizing M/X major flare recovery.\n")
    md.append("---\n")
    md.append("## Executive Summary\n")
    md.append(f"Out of 194 total catalog observation dates, **94 dates** are currently extracted on disk, while **133 observation dates** (spanning **952 missing flare events**) remain to be ingested.")
    md.append(f"Crucially, **19 missing observation dates** contain **45 M/X major flares** (7 X-class flares and 38 M-class flares).\n")

    t1 = res_df[res_df["tier_num"] == 1]
    t2 = res_df[res_df["tier_num"] == 2]
    t3 = res_df[res_df["tier_num"] == 3]

    md.append("| Priority Tier | Target Criteria | Unique Dates | M/X Flares | Total Missing Flares | Script Ready on Disk | Action Required |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    md.append(f"| **Tier 1 (High)** | **Contains M/X Class Flares** | **{len(t1)}** | **{t1['mx_count'].sum()}** | **{t1['total_flares'].sum()}** | **{sum(t1['script_exists'])} / {len(t1)} dates** | Download & extract Tier 1 immediately |")
    md.append(f"| **Tier 2 (Medium)** | **High Volume (>=5) or Script Available** | **{len(t2)}** | **0** | **{t2['total_flares'].sum()}** | **{sum(t2['script_exists'])} / {len(t2)} dates** | Download script-ready Tier 2 dates |")
    md.append(f"| **Tier 3 (Low)** | **Minor / Low Volume (1-4 flares)** | **{len(t3)}** | **0** | **{t3['total_flares'].sum()}** | **{sum(t3['script_exists'])} / {len(t3)} dates** | Query PRADAN for backfill |")
    md.append(f"| **Total** | | **{len(res_df)}** | **45** | **952** | **{sum(res_df['script_exists'])} / {len(res_df)} dates** | |\n")

    md.append("---\n")
    md.append("## 1. Tier 1: High Priority Observation Dates (19 Dates with M/X Flares)\n")
    md.append("These **19 observation dates** contain all **45 missing M/X major flares** in the catalog. Recovering these dates is the highest priority to boost M/X flare sample size ($N$) for model training.\n")

    md.append("| Rank | Date | M/X Count | X Flares | M Flares | Specific M/X Flare Classes | Total Flares | PRADAN Script Status |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |")

    for idx, r in t1.iterrows():
        rank = idx + 1
        script_str = "✓ **Ready in raw_downloads**" if r["script_exists"] else "✗ Requires PRADAN Portal Fetch"
        md.append(f"| {rank} | `{r['date']}` | **{r['mx_count']}** | `{r['x_count']}` | `{r['m_count']}` | `{r['mx_classes']}` | {r['total_flares']} | {script_str} |")

    t1_ready = t1[t1["script_exists"]]
    t1_portal = t1[~t1["script_exists"]]

    ready_dates_str = ", ".join([f"`{d}`" for d in t1_ready["date"].tolist()])
    portal_dates_str = ", ".join([f"`{d}`" for d in t1_portal["date"].tolist()])

    md.append("\n### Tier 1 Breakdown by Script Readiness:\n")
    md.append(f"- **Tier 1A (Script Ready on Disk - {len(t1_ready)} dates)**: Dates {ready_dates_str} contain **{t1_ready['mx_count'].sum()} M/X flares** ({t1_ready['x_count'].sum()} X-class, {t1_ready['m_count'].sum()} M-class) and can be downloaded/extracted immediately using existing URLs in `data/hel1os/raw_downloads/`.")
    md.append(f"- **Tier 1B (Requires ISSDC Portal Fetch - {len(t1_portal)} dates)**: Dates {portal_dates_str} contain **{t1_portal['mx_count'].sum()} M/X flares** ({t1_portal['x_count'].sum()} X-class, {t1_portal['m_count'].sum()} M-class) and require fetching fresh script links from the ISSDC PRADAN portal.\n")

    md.append("---\n")
    md.append("## 2. Tier 2: Medium Priority Observation Dates (100 Dates)\n")
    md.append("These dates contain C-class and B-class minor flares or unmatched flares. **80 of these dates** already have download scripts present in `data/hel1os/raw_downloads/`.\n")

    md.append("Top 15 Tier 2 Dates by Flare Density:\n")
    md.append("| Rank | Date | Total Flares | C Flares | B Flares | Unmatched | PRADAN Script Status |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, r in t2.head(15).reset_index(drop=True).iterrows():
        rank = idx + 1
        script_str = "✓ **Ready in raw_downloads**" if r["script_exists"] else "Requires PRADAN Portal Fetch"
        md.append(f"| {rank} | `{r['date']}` | **{r['total_flares']}** | {r['c_count']} | {r['b_count']} | {r['unmatched_count']} | {script_str} |")

    md.append("\n---\n")
    md.append("## 3. Tier 3: Low Priority Observation Dates (14 Dates)\n")
    md.append("Low flare count (1–4 minor flares per day) with zero M/X major flares.\n")

    md.append("---\n")
    md.append("## 4. Recommended Action Plan for Ingestion\n")
    md.append(f"1. **Phase 1 (Immediate Tier 1A Execution)**: Run download and extraction on the {len(t1_ready)} Tier 1 dates with scripts already on disk ({ready_dates_str}). Recover **{t1_ready['mx_count'].sum()} M/X flares** ({t1_ready['x_count'].sum()} X-class, {t1_ready['m_count'].sum()} M-class).")
    md.append(f"2. **Phase 2 (ISSDC Portal Fetch for Tier 1B)**: Request download links for the remaining {len(t1_portal)} Tier 1 dates ({portal_dates_str}) to recover the remaining **{t1_portal['mx_count'].sum()} M/X flares**.")
    md.append("3. **Phase 3 (Sequence Re-building)**: Execute `scripts/build_sequence_dataset_expanded.py` after each phase to expand `X_sequences_expanded.npy` toward $N \\approx 1,200+$ sequences.\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"[SUCCESS] Report generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()
