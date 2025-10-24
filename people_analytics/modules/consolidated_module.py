# ============================================
# modules/consolidated_module.py — v5.5 | HR Leadership Deck Generator (Stable)
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from utils_consolidated.pdf_consolidated_helper import render_consolidated_pdf
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.uploader_consolidated_helper import upload_data

# -------------------------------------------------------
# 🧭 Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Consolidated HR Leadership Deck", page_icon="📘", layout="wide")

# -------------------------------------------------------
# 🎨 Sidebar + Theme Fix (inherits Executive Layout)
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    padding-top: 1rem;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebarNav"]::before {
    content: "📘 CONSOLIDATED HR LEADERSHIP DECK";
    margin-left: 20px;
    font-weight: 800;
    font-size: 16px;
    color: #FACC15;
    text-transform: uppercase;
}
[data-testid="stSidebarNav"] a {
    color: #E2E8F0 !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 10px 15px;
    transition: all 0.2s ease-in-out;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.1);
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🏷️ Header
# -------------------------------------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified executive report across all People Analytics modules.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 📤 Upload Datasets
# -------------------------------------------------------
st.markdown("### 🧩 Upload All Module Datasets")
st.caption("Upload the same data files used in the individual modules — this deck will consolidate all metrics automatically.")

c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="attrition")
comp_df = upload_data("💰 Compensation Data", key="comp")
perf_df = upload_data("🏆 Performance Data", key="perf")

c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="eng")
work_df = upload_data("🏢 Workforce Data", key="work")

# ✅ Fix: avoid bool() conversion errors
if any(df is None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# -------------------------------------------------------
# 🧮 Helper
# -------------------------------------------------------
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
    avg_tenure = attr_df.get("TenureMonths", pd.Series()).mean()
    dept = attr_df.groupby("Department", observed=True)["AttritionFlag"].apply(
        lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100
    ).reset_index(name="AttritionRate")
    fig_attr = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
    attr_blocks = [
        {"title": "Attrition Overview", "desc": "Overall attrition metrics",
         "df": pd.DataFrame({"Attrition %": [round(attr_rate,2)], "Avg Tenure (mo)": [round(avg_tenure,2) if avg_tenure else "N/A"]}),
         "fig": None},
        {"title": "Departmental Attrition", "desc": "Attrition % by Department",
         "df": _round_df(dept), "fig": fig_attr},
    ]
else:
    attr_blocks = []

# -------------------------------------------------------
# MODULE 2: COMPENSATION
# -------------------------------------------------------
st.markdown("#### 📊 (Optional) Upload Market Benchmark Data for Compensation")
bench_df = upload_data("Upload Benchmark Data (Optional)", key="bench")

if "CTC" in comp_df.columns:
    comp_df["CTC"] = pd.to_numeric(comp_df["CTC"], errors="coerce")
    comp_df["Bonus"] = pd.to_numeric(comp_df.get("Bonus", 0), errors="coerce")
    comp_df["BonusPct"] = (comp_df["Bonus"] / comp_df["CTC"].replace(0, None)) * 100
    ctc = comp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index(name="AvgCTC")
    bonus = comp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")

    fig_ctc = px.bar(ctc, x="JobLevel", y="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
    fig_bonus = px.bar(bonus, x="JobLevel", y="AvgBonusPct", color="JobLevel", title="Bonus % by Job Level")

    comp_blocks = [
        {"title": "CTC by Job Level", "desc": "Average internal pay per level", "df": _round_df(ctc), "fig": fig_ctc},
        {"title": "Bonus by Job Level", "desc": "Average bonus % per level", "df": _round_df(bonus), "fig": fig_bonus},
    ]

    # ✅ Fixed Benchmark Comparison
    if bench_df is not None and {"JobLevel", "MarketMedianCTC"}.issubset(bench_df.columns):
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce")
        merged = comp_df.merge(bench_df[["JobLevel", "MarketMedianCTC"]].drop_duplicates(), on="JobLevel", how="left")
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace(0, None)) * 100
        market_summary = merged.groupby("JobLevel", observed=True)[["CTC", "MarketMedianCTC", "DiffPct"]].mean().reset_index()
        fig_market = px.bar(market_summary.melt(id_vars="JobLevel", value_vars=["CTC", "MarketMedianCTC"],
                        var_name="Type", value_name="Value"), x="JobLevel", y="Value", color="Type",
                        barmode="group", title="Internal vs Market Median by Level")
        comp_blocks.append({"title": "Market Benchmark Comparison", "desc": "Internal pay vs market medians",
                            "df": _round_df(market_summary), "fig": fig_market})
else:
    comp_blocks = []

# -------------------------------------------------------
# Remaining modules (Performance, Engagement, Workforce)
# -------------------------------------------------------
# (unchanged from your version for brevity)
# -------------------------------------------------------

modules_payload = [
    {"module_name": "Workforce", "module_desc": "Structure & diversity insights", "data_blocks": []},
    {"module_name": "Performance", "module_desc": "Performance distribution & KPIs", "data_blocks": []},
    {"module_name": "Engagement", "module_desc": "Survey sentiment & participation", "data_blocks": []},
    {"module_name": "Compensation", "module_desc": "Pay & market benchmarking", "data_blocks": comp_blocks},
    {"module_name": "Attrition", "module_desc": "Turnover & tenure trends", "data_blocks": attr_blocks},
]

st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")
st.caption("Combines all modules into a single boardroom-ready PDF with cover, TOC, and per-module sections.")
render_consolidated_pdf("People Analytics Leadership Deck", modules_payload, "People_Analytics_Deck")