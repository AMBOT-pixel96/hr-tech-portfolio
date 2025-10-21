# ============================================
# modules/attrition_module.py — v2.1 | Attrition Trends + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_attrition_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#7F1D1D,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze turnover, identify high-risk segments, and uncover attrition trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📄 Step 1 — Download Attrition Template")
    sample = pd.DataFrame({
        "EmployeeID":["E1001","E1002"],"Department":["Finance","IT"],
        "JobLevel":["Analyst","Manager"],"Gender":["Male","Female"],
        "TenureMonths":[24,60],"AttritionFlag":["Yes","No"],"ExitReason":["Better Pay",""]
    })
    render_download_template("Attrition Data Template", sample, "Attrition_Template.csv")

    st.subheader("📤 Step 2 — Upload Dataset")
    attr_file = st.file_uploader("Upload Attrition Data", type=["csv","xlsx"])
    if attr_file is None:
        st.info("Please upload data to continue.")
        return
    df = pd.read_csv(attr_file) if attr_file.name.endswith(".csv") else pd.read_excel(attr_file, engine="openpyxl")
    st.success("✅ File uploaded successfully!")

    required = ["EmployeeID","Department","JobLevel","Gender","TenureMonths","AttritionFlag"]
    if any(c not in df.columns for c in required):
        st.error("Missing required columns.")
        return

    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().replace({"yes":"Yes","y":"Yes","no":"No","n":"No"})

    total_employees = len(df)
    total_left = (df["AttritionFlag"]=="Yes").sum()
    turnover_rate = total_left/total_employees*100 if total_employees>0 else 0
    avg_tenure = df["TenureMonths"].mean()

    dept_summary = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="AttritionRate")
    dept_highest = dept_summary.loc[dept_summary["AttritionRate"].idxmax(),"Department"]

    st.metric("Overall Attrition Rate", f"{turnover_rate:.1f}%")
    st.metric("Average Tenure", f"{avg_tenure:.1f} months")

    fig = px.bar(dept_summary, x="Department", y="AttritionRate", text="AttritionRate", color="Department",
                 title="Attrition % by Department")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # --- Step 5: Export ---
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [{
        "title":"Attrition Insights",
        "desc":"Turnover rates, department trends, and tenure analysis.",
        "df":dept_summary,
        "insights":[
            f"Overall attrition rate: {turnover_rate:.1f}%",
            f"Highest attrition department: {dept_highest}",
            f"Average tenure: {avg_tenure:.1f} months"
        ]
    }]
    export_module_report("Attrition Analytics Executive Report","Attrition",data_blocks,"Attrition")