# ============================================
# utils_consolidated/pdf_helper_consolidated.py — v1.0 | For Consolidated Deck Only
# ============================================
import json, os

def extract_summary_table(tmp_dir):
    """Extracts key insights & metric names from each module JSON for consolidated summary."""
    summaries = []
    for file in os.listdir(tmp_dir):
        if file.endswith(".json"):
            module_name = file.replace(".json", "")
            with open(os.path.join(tmp_dir, file), "r", encoding="utf-8") as f:
                meta = json.load(f)
            summaries.append({
                "Module": module_name,
                "Metrics": meta.get("metrics_short", "N/A"),
                "Insights": meta.get("insights", "No summary provided.")
            })
    return summaries