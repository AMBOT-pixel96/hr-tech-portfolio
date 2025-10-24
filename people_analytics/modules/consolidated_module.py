# ============================================
# modules/consolidated_module.py — v5.8 | Uncrashable HR Leadership Deck Generator
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
import traceback, time, os

from utils_consolidated.pdf_consolidated_helper import render_consolidated_pdf
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.uploader_consolidated_helper import upload_data

# -------------------------------------------------------
# 🎨 Header
# -------------------------------------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified executive report across all People Analytics modules.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 📤 Upload Section
# -------------------------------------------------------
st.markdown("### 🧩 Upload All Module Datasets")
c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="attrition")
comp_df = upload_data("💰 Compensation Data", key="comp")
perf_df = upload_data("🏆 Performance Data", key="perf")
c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="eng")
work_df = upload_data("🏢 Workforce Data", key="work")

# ✅ Input Validation
if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# -------------------------------------------------------
# 🧮 Helper
# -------------------------------------------------------
def _round_df(df, decimals=2, limit=25):
    if df is None or df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2.head(limit)

# -------------------------------------------------------
# ⚙️ Safe Section Wrapper
# -------------------------------------------------------
def safe_block(label, func):
    try:
        with st.spinner(f"⚙️ Processing {label}..."):
            out = func()
            if not out:
                st.warning(f"⚠️ {label} skipped (no data).")
            return out
    except Exception as e:
        st.error(f"❌ Error in {label}: {e}")
        st.code(traceback.format_exc())
        return []

# -------------------------------------------------------
# MODULES
# -------------------------------------------------------
def mod_attrition():
    if "AttritionFlag" not in attr_df.columns:
        return []
    attr_rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean()) * 100
    avg_tenure = attr_df["TenureMonths"].mean() if "TenureMonths" in attr_df else None
    dept = attr_df.groupby("Department", observed=True)["AttritionFlag"].apply(
        lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100
    ).reset_index(name="AttritionRate")
    fig = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
    return [
        {"title":"Attrition Overview","desc":"Overall attrition metrics",
         "df":pd.DataFrame({"Attrition %":[round(attr_rate,2)],"Avg Tenure (mo)":[round(avg_tenure,2) if avg_tenure else "N/A"]}),
         "fig":None,"insights":[f"Attrition rate: {attr_rate:.1f}%",f"Avg tenure: {avg_tenure:.1f} mo" if avg_tenure else "N/A"]},
        {"title":"Departmental Attrition","desc":"Attrition % by Department","df":_round_df(dept),"fig":fig,"insights":[]}
    ]

def mod_compensation():
    if "CTC" not in comp_df.columns:
        return []
    comp_df["CTC"]=pd.to_numeric(comp_df["CTC"],errors="coerce")
    comp_df["Bonus"]=pd.to_numeric(comp_df.get("Bonus",0),errors="coerce")
    comp_df["BonusPct"]=(comp_df["Bonus"]/comp_df["CTC"].replace(0,pd.NA))*100
    ctc=comp_df.groupby("JobLevel",observed=True)["CTC"].mean().reset_index(name="AvgCTC")
    bonus=comp_df.groupby("JobLevel",observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")
    fig_ctc=px.bar(ctc,x="JobLevel",y="AvgCTC",color="JobLevel",title="Avg CTC by Job Level")
    fig_bonus=px.bar(bonus,x="JobLevel",y="AvgBonusPct",color="JobLevel",title="Bonus % by Job Level")
    return [
        {"title":"CTC by Job Level","desc":"Average pay per level","df":_round_df(ctc),"fig":fig_ctc},
        {"title":"Bonus by Job Level","desc":"Average bonus % per level","df":_round_df(bonus),"fig":fig_bonus}
    ]

def mod_performance():
    if "PerformanceRating" not in perf_df.columns:
        return []
    perf_df["PerformanceRating"]=pd.to_numeric(perf_df["PerformanceRating"],errors="coerce")
    job_perf=perf_df.groupby("JobLevel",observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
    fig=px.bar(job_perf,x="JobLevel",y="AvgRating",color="JobLevel",title="Avg Performance Rating by Job Level")
    return [{"title":"Performance Summary","desc":"Average rating per job level","df":_round_df(job_perf),"fig":fig}]

def mod_engagement():
    qcols=[c for c in eng_df.columns if c.upper().startswith("Q")]
    if not qcols: return []
    eng_df[qcols]=eng_df[qcols].apply(pd.to_numeric,errors="coerce")
    eng_df["EngagementIndex"]=eng_df[qcols].mean(axis=1)
    dept_eng=eng_df.groupby("Department",observed=True)["EngagementIndex"].mean().reset_index(name="MeanIndex")
    fig=px.bar(dept_eng,x="Department",y="MeanIndex",color="Department",title="Engagement Index by Department")
    return [
        {"title":"Engagement Overview","desc":"Overall engagement index","df":pd.DataFrame({"Average Index":[eng_df['EngagementIndex'].mean().round(2)]}),"fig":None},
        {"title":"Departmental Engagement","desc":"Avg engagement by department","df":_round_df(dept_eng),"fig":fig}
    ]

def mod_workforce():
    if "JobLevel" not in work_df.columns:
        return []
    headcount=work_df.groupby("JobLevel",observed=True).size().reset_index(name="Headcount")
    gender_split=work_df["Gender"].value_counts(normalize=True).mul(100).reset_index()
    gender_split.columns=["Gender","Percent"]
    fig_hc=px.bar(headcount,x="JobLevel",y="Headcount",color="JobLevel",title="Headcount by Job Level")
    fig_gender=px.pie(gender_split,names="Gender",values="Percent",title="Gender Composition")
    return [
        {"title":"Headcount Structure","desc":"Employee count by level","df":_round_df(headcount),"fig":fig_hc},
        {"title":"Gender Composition","desc":"Gender % across workforce","df":_round_df(gender_split),"fig":fig_gender}
    ]

# -------------------------------------------------------
# 🧩 Consolidate & Render
# -------------------------------------------------------
modules_payload=[
    {"module_name":"Workforce","module_desc":"Structure & diversity insights","data_blocks":safe_block("Workforce",mod_workforce)},
    {"module_name":"Performance","module_desc":"Performance distribution & KPIs","data_blocks":safe_block("Performance",mod_performance)},
    {"module_name":"Engagement","module_desc":"Engagement & sentiment trends","data_blocks":safe_block("Engagement",mod_engagement)},
    {"module_name":"Compensation","module_desc":"Pay & incentive analytics","data_blocks":safe_block("Compensation",mod_compensation)},
    {"module_name":"Attrition","module_desc":"Turnover & tenure insights","data_blocks":safe_block("Attrition",mod_attrition)},
]

st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")
st.caption("Combines all modules into a single polished PDF.")

# 🧠 Safety wrapper around PDF build
try:
    if st.button("🧾 Generate Consolidated Leadership Deck", use_container_width=True):
        st.info("🕓 Building deck... please wait 10–15 seconds.")
        render_consolidated_pdf("People Analytics Leadership Deck", modules_payload, "People_Analytics_Deck")
        st.success("✅ Deck built successfully.")
except Exception as e:
    st.error("💀 Fatal error while generating PDF.")
    st.code(traceback.format_exc())
    st.stop()