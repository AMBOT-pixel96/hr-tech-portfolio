# ============================================
# modules/consolidated_module.py — v2.0 | Clean Build (WeasyPrint Edition)
# ============================================
import os
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

from utils_consolidated.uploader_consolidated_helper import upload_data
from utils_consolidated.pdf_consolidated_helper import render_consolidated_pdf  # ✅ new safe builder

# ----------------------------
# Logging / diagnostics
# ----------------------------
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)
LOG_PATH = os.path.join(TMP_DIR, "consolidated_build.log")

def log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except Exception:
        pass

# ----------------------------
# Page header (UI)
# ----------------------------
st.markdown("""
<div style="padding:14px;border-radius:8px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Upload module data and generate a boardroom-ready PDF (no Kaleido, no temp writes).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🧩 Upload All Module Datasets")
st.caption("Supported formats: CSV, XLS, XLSX. Max recommended size per file: 200MB")

# ----------------------------
# Upload inputs
# ----------------------------
c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="con_attr")
comp_df = upload_data("💰 Compensation Data", key="con_comp")
perf_df = upload_data("🏆 Performance Data", key="con_perf")

c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="con_eng")
work_df = upload_data("🏢 Workforce Data", key="con_work")

if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

# ----------------------------
# Build data blocks
# ----------------------------
log("Preparing module payloads")
try:
    # ATTRITION
    attr_blocks = []
    if "AttritionFlag" in attr_df.columns:
        attr_rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean()) * 100
        avg_tenure = attr_df["TenureMonths"].mean() if "TenureMonths" in attr_df else None
        dept = (
            attr_df.groupby("Department", observed=True)["AttritionFlag"]
            .apply(lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig_attr = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
        attr_blocks = [
            {"title": "Attrition Overview", "desc": "Overall attrition metrics",
             "df": pd.DataFrame({"Attrition %": [round(attr_rate,2)], "Avg Tenure (mo)": [round(avg_tenure,2) if avg_tenure else "N/A"]}),
             "fig": None, "insights": [f"Attrition rate: {attr_rate:.1f}%", f"Avg tenure: {avg_tenure:.1f} mo" if avg_tenure else "N/A"]},
            {"title": "Departmental Attrition", "desc": "Attrition % by Department",
             "df": _round_df(dept), "fig": fig_attr, "insights": []},
        ]

    # COMPENSATION
    comp_blocks = []
    if "CTC" in comp_df.columns:
        comp_df["CTC"] = pd.to_numeric(comp_df["CTC"], errors="coerce")
        comp_df["Bonus"] = pd.to_numeric(comp_df.get("Bonus", 0), errors="coerce")
        comp_df["BonusPct"] = (comp_df["Bonus"] / comp_df["CTC"].replace(0, pd.NA)) * 100
        ctc = comp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index(name="AvgCTC")
        bonus = comp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")
        fig_ctc = px.bar(ctc, x="JobLevel", y="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
        fig_bonus = px.bar(bonus, x="JobLevel", y="AvgBonusPct", color="JobLevel", title="Bonus % by Job Level")
        comp_blocks = [
            {"title": "CTC by Job Level", "desc": "Average internal pay per level", "df": _round_df(ctc), "fig": fig_ctc},
            {"title": "Bonus by Job Level", "desc": "Average bonus % per level", "df": _round_df(bonus), "fig": fig_bonus},
        ]

    # PERFORMANCE
    perf_blocks = []
    if "PerformanceRating" in perf_df.columns:
        perf_df["PerformanceRating"] = pd.to_numeric(perf_df["PerformanceRating"], errors="coerce")
        job_perf = perf_df.groupby("JobLevel", observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
        fig_perf = px.bar(job_perf, x="JobLevel", y="AvgRating", color="JobLevel", title="Avg Performance Rating by Job Level")
        perf_blocks = [{"title": "Performance Summary", "desc": "Average rating per job level",
                        "df": _round_df(job_perf), "fig": fig_perf}]

    # ENGAGEMENT
    eng_blocks = []
    qcols = [c for c in eng_df.columns if c.upper().startswith("Q")]
    if qcols:
        eng_df[qcols] = eng_df[qcols].apply(pd.to_numeric, errors="coerce")
        eng_df["EngagementIndex"] = eng_df[qcols].mean(axis=1)
        dept_eng = eng_df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index(name="MeanIndex")
        fig_eng = px.bar(dept_eng, x="Department", y="MeanIndex", color="Department", title="Engagement Index by Department")
        eng_blocks = [
            {"title": "Engagement Overview", "desc": "Overall engagement index",
             "df": pd.DataFrame({"Average Index": [eng_df['EngagementIndex'].mean().round(2)]}), "fig": None},
            {"title": "Departmental Engagement", "desc": "Avg engagement by department",
             "df": _round_df(dept_eng), "fig": fig_eng},
        ]

    # WORKFORCE
    work_blocks = []
    if "JobLevel" in work_df.columns:
        headcount = work_df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
        fig_hc = px.bar(headcount, x="JobLevel", y="Headcount", color="JobLevel", title="Headcount by Job Level")
        gender_split = work_df["Gender"].value_counts(normalize=True).mul(100).reset_index()
        gender_split.columns = ["Gender", "Percent"]
        fig_gender = px.pie(gender_split, names="Gender", values="Percent", title="Gender Composition")
        work_blocks = [
            {"title": "Headcount Structure", "desc": "Employee count by level", "df": headcount, "fig": fig_hc},
            {"title": "Gender Composition", "desc": "Gender % across workforce", "df": gender_split, "fig": fig_gender},
        ]

    modules_payload = [
        {"module_name": "Attrition", "module_desc": "Turnover & tenure trends", "data_blocks": attr_blocks},
        {"module_name": "Compensation", "module_desc": "Pay & benchmarking", "data_blocks": comp_blocks},
        {"module_name": "Performance", "module_desc": "Performance distribution & KPIs", "data_blocks": perf_blocks},
        {"module_name": "Engagement", "module_desc": "Survey sentiment & participation", "data_blocks": eng_blocks},
        {"module_name": "Workforce", "module_desc": "Structure & diversity insights", "data_blocks": work_blocks},
    ]

except Exception as e:
    log(f"Payload prep error: {e}\n{traceback.format_exc()}")
    st.error("⚠️ Failed preparing payloads. Check /tmp/consolidated_build.log")
    st.stop()

# ----------------------------
# Generate PDF (WeasyPrint)
# ----------------------------
st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")
st.caption("Combines all uploaded modules into a single PDF (no file I/O).")

render_consolidated_pdf("People Analytics Leadership Deck", modules_payload, "People_Analytics_Deck")