# ============================================
# modules/consolidated_module.py — v5.2 | HR Leadership Deck Generator (Boardroom Edition)
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px

from utils_consolidated.pdf_consolidated_helper import render_consolidated_pdf
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved

# -------------------------------------------------------
# 🎨 HEADER UI (with branding tag)
# -------------------------------------------------------
st.markdown("""
<div style="position:relative;padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0;">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified executive report across all People Analytics modules.</p>
  <div style="position:absolute;top:18px;right:25px;font-size:13px;color:#93C5FD;font-weight:600;">
    People Analytics | v5.2 Boardroom Edition
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 💅 STYLING (Streamlit CSS Theme)
# -------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'DejaVu Sans', sans-serif;
    color: #111827;
}

/* ===== Header Banner ===== */
div[data-testid="stMarkdownContainer"] h2 {
    font-size: 26px !important;
    color: white !important;
    text-shadow: 0px 0px 6px rgba(0,0,0,0.25);
}

/* ===== Uploaders ===== */
div.stFileUploader, div[data-testid="stFileUploaderDropzone"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.08);
}

/* ===== Buttons ===== */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #1E3A8A, #2563EB);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #2563EB, #1E40AF);
    transform: scale(1.02);
}

/* ===== Toasts & Boxes ===== */
div[data-baseweb="toast"] {
    font-size: 13px !important;
    border-radius: 10px !important;
    padding: 6px !important;
}

/* ===== PDF Button Glow ===== */
button[kind="primary"] {
    box-shadow: 0px 3px 8px rgba(37,99,235,0.3);
}

/* ===== Hide Streamlit Footer ===== */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 📂 UPLOAD SECTION
# -------------------------------------------------------
st.markdown("### 🧩 Upload All Module Datasets")
st.caption("Upload the same data files used in individual modules — this deck will consolidate all metrics automatically.")

c1, c2, c3 = st.columns(3)
attr_file = c1.file_uploader("📉 Attrition Data", type=["csv", "xlsx"])
comp_file = c2.file_uploader("💰 Compensation Data", type=["csv", "xlsx"])
perf_file = c3.file_uploader("🏆 Performance Data", type=["csv", "xlsx"])

c4, c5 = st.columns(2)
eng_file = c4.file_uploader("💬 Engagement Data", type=["csv", "xlsx"])
work_file = c5.file_uploader("🏢 Workforce Data", type=["csv", "xlsx"])

if not all([attr_file, comp_file, perf_file, eng_file, work_file]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# -------------------------------------------------------
# 📊 LOADER HELPER
# -------------------------------------------------------
def load_data(file):
    """Universal loader with Excel + CSV support."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        st.error(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame()

# -------------------------------------------------------
# 🧠 LOAD DATASETS
# -------------------------------------------------------
attr_df = load_data(attr_file)
comp_df = load_data(comp_file)
perf_df = load_data(perf_file)
eng_df = load_data(eng_file)
work_df = load_data(work_file)

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

# -------------------------------------------------------
# MODULE 1: ATTRITION
# -------------------------------------------------------
if "AttritionFlag" in attr_df.columns:
    attr_rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean()) * 100
    avg_tenure = attr_df["TenureMonths"].mean() if "TenureMonths" in attr_df else None
    dept = attr_df.groupby("Department", observed=True)["AttritionFlag"].apply(
        lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100
    ).reset_index(name="AttritionRate")
    fig_attr = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")

    attr_blocks = [
        {
            "title": "Attrition Overview",
            "desc": "Overall attrition metrics",
            "df": pd.DataFrame({
                "Attrition %": [round(attr_rate,2)],
                "Avg Tenure (mo)": [round(avg_tenure,2) if avg_tenure else "N/A"]
            }),
            "fig": None,
            "insights": [f"Attrition rate: {attr_rate:.1f}%", f"Avg tenure: {avg_tenure:.1f} mo" if avg_tenure else "N/A"]
        },
        {"title": "Departmental Attrition", "desc": "Attrition % by Department", "df": _round_df(dept), "fig": fig_attr, "insights": []},
    ]
else:
    attr_blocks = []

# -------------------------------------------------------
# MODULE 2: COMPENSATION
# -------------------------------------------------------
bench_df = None
st.markdown("#### 📊 (Optional) Upload Market Benchmark Data for Compensation")
bench_file = st.file_uploader("Upload Benchmark Data (Optional)", type=["csv", "xlsx"], key="bench_upload")
if bench_file:
    try:
        bench_df = load_data(bench_file)
        st.success(f"✅ {bench_file.name} uploaded successfully (Benchmark Data)")
    except Exception as e:
        st.warning(f"⚠️ Failed to load benchmark data: {e}")

if "CTC" in comp_df.columns:
    comp_df["CTC"] = pd.to_numeric(comp_df["CTC"], errors="coerce")
    comp_df["Bonus"] = pd.to_numeric(comp_df.get("Bonus", 0), errors="coerce")
    comp_df["BonusPct"] = (comp_df["Bonus"] / comp_df["CTC"].replace(0, None)) * 100

    ctc = comp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index(name="AvgCTC")
    bonus = comp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")

    fig_ctc = px.bar(ctc, x="JobLevel", y="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
    fig_bonus = px.bar(bonus, x="JobLevel", y="AvgBonusPct", color="JobLevel", title="Bonus % by Job Level")

    comp_blocks = [
        {"title": "CTC by Job Level", "desc": "Average internal pay per level", "df": _round_df(ctc), "fig": fig_ctc, "insights": []},
        {"title": "Bonus by Job Level", "desc": "Average bonus % per level", "df": _round_df(bonus), "fig": fig_bonus, "insights": []},
    ]

    # Optional benchmark
    if bench_df is not None and {"JobLevel", "MarketMedianCTC"}.issubset(set(bench_df.columns)):
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce")
        merged = comp_df.merge(bench_df[["JobLevel", "MarketMedianCTC"]].drop_duplicates(), on="JobLevel", how="left")
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace(0, None)) * 100
        market_summary = merged.groupby("JobLevel", observed=True)[["CTC", "MarketMedianCTC", "DiffPct"]].mean().reset_index()
        market_summary = _round_df(market_summary)
        fig_market = px.bar(market_summary.melt(id_vars="JobLevel", value_vars=["CTC", "MarketMedianCTC"],
                            var_name="Type", value_name="Value"), x="JobLevel", y="Value", color="Type",
                            barmode="group", title="Internal vs Market Median by Level")
        comp_blocks.append({"title": "Market Benchmark Comparison", "desc": "Internal pay vs market medians",
                            "df": market_summary, "fig": fig_market, "insights": []})
else:
    comp_blocks = []

# -------------------------------------------------------
# MODULE 3: PERFORMANCE
# -------------------------------------------------------
if "PerformanceRating" in perf_df.columns:
    perf_df["PerformanceRating"] = pd.to_numeric(perf_df["PerformanceRating"], errors="coerce")
    job_perf = perf_df.groupby("JobLevel", observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
    fig_perf = px.bar(job_perf, x="JobLevel", y="AvgRating", color="JobLevel", title="Avg Performance Rating by Job Level")
    perf_blocks = [
        {"title": "Performance Summary", "desc": "Average rating per job level", "df": _round_df(job_perf), "fig": fig_perf,
         "insights": [f"Overall avg rating: {perf_df['PerformanceRating'].mean():.2f}"]},
    ]
else:
    perf_blocks = []

# -------------------------------------------------------
# MODULE 4: ENGAGEMENT
# -------------------------------------------------------
qcols = [c for c in eng_df.columns if c.upper().startswith("Q")]
if qcols:
    eng_df[qcols] = eng_df[qcols].apply(pd.to_numeric, errors="coerce")
    eng_df["EngagementIndex"] = eng_df[qcols].mean(axis=1)
    dept_eng = eng_df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index(name="MeanIndex")
    fig_eng = px.bar(dept_eng, x="Department", y="MeanIndex", color="Department", title="Engagement Index by Department")
    eng_blocks = [
        {"title": "Engagement Overview", "desc": "Overall engagement index", "df": pd.DataFrame({"Average Index": [eng_df['EngagementIndex'].mean().round(2)]}), "fig": None,
         "insights": [f"Avg engagement index: {eng_df['EngagementIndex'].mean():.2f}"]},
        {"title": "Departmental Engagement", "desc": "Avg engagement by department", "df": _round_df(dept_eng), "fig": fig_eng, "insights": []},
    ]
else:
    eng_blocks = []

# -------------------------------------------------------
# MODULE 5: WORKFORCE
# -------------------------------------------------------
if "JobLevel" in work_df.columns:
    headcount = work_df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    fig_hc = px.bar(headcount, x="JobLevel", y="Headcount", color="JobLevel", title="Headcount by Job Level")
    gender_split = work_df["Gender"].value_counts(normalize=True).mul(100).reset_index()
    gender_split.columns = ["Gender", "Percent"]
    fig_gender = px.pie(gender_split, names="Gender", values="Percent", title="Gender Composition")
    work_blocks = [
        {"title": "Headcount Structure", "desc": "Employee count by level", "df": headcount, "fig": fig_hc, "insights": []},
        {"title": "Gender Composition", "desc": "Gender % across workforce", "df": gender_split, "fig": fig_gender, "insights": []},
    ]
else:
    work_blocks = []

# -------------------------------------------------------
# 🧩 CONSOLIDATE MODULES
# -------------------------------------------------------
modules_payload = [
    {"module_name": "Attrition", "module_desc": "Turnover & tenure trends", "data_blocks": attr_blocks},
    {"module_name": "Compensation", "module_desc": "Pay and incentive analytics", "data_blocks": comp_blocks},
    {"module_name": "Performance", "module_desc": "Performance distribution & KPIs", "data_blocks": perf_blocks},
    {"module_name": "Engagement", "module_desc": "Survey sentiment & participation", "data_blocks": eng_blocks},
    {"module_name": "Workforce", "module_desc": "Structure & diversity insights", "data_blocks": work_blocks},
]

# -------------------------------------------------------
# 📄 GENERATE PDF
# -------------------------------------------------------
st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")
st.caption("Combines all modules into a single boardroom-ready PDF with cover, TOC, and per-module sections.")

with st.spinner("🧠 Generating your HR Leadership Deck..."):
    render_consolidated_pdf("People Analytics Leadership Deck", modules_payload, "People_Analytics_Deck")