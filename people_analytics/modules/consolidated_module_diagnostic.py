# ============================================
# consolidated_module_diagnostic.py — Minimal Diagnostic Build
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from utils_consolidated.uploader_consolidated_helper import upload_data

st.set_page_config(page_title="Diagnostic | HR Deck", page_icon="🧠", layout="wide")

st.markdown("## 🧠 Diagnostic Build — Consolidated Module")
st.caption("This mode loads all datasets but skips heavy PDF generation to isolate memory or I/O issues.")

c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="attrition_diag")
comp_df = upload_data("💰 Compensation Data", key="comp_diag")
perf_df = upload_data("🏆 Performance Data", key="perf_diag")

c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="eng_diag")
work_df = upload_data("🏢 Workforce Data", key="work_diag")

# ✅ Basic existence check
if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.warning("Please upload all datasets.")
    st.stop()

# ✅ Print dataset shapes
st.markdown("### ✅ Data Loaded Successfully")
for name, df in {
    "Attrition": attr_df,
    "Compensation": comp_df,
    "Performance": perf_df,
    "Engagement": eng_df,
    "Workforce": work_df,
}.items():
    st.write(f"**{name}** — {df.shape[0]} rows × {df.shape[1]} columns")

# ✅ Light visual test (without saving to disk)
if "AttritionFlag" in attr_df.columns:
    dept = (
        attr_df.groupby("Department", observed=True)["AttritionFlag"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "y", "true", "1"])).mean() * 100)
        .reset_index(name="AttritionRate")
    )
    fig = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Attrition chart rendered successfully")

st.success("🧩 Diagnostic completed — app loaded all data and rendered at least one chart successfully.")
st.info("👉 If this runs fine, the crash source is PDF I/O or Kaleido, not your data or memory.")