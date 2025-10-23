import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image

def run_attrition_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#7F1D1D,#DC2626);color:white;">
      <h2 style="margin:0">📉 Attrition Analytics</h2>
      <p style="margin:4px 0 0 0;">Turnover, tenure cohorts & exit reasons (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Attrition Data (CSV/XLSX)")
    if df is None:
        return

    if "TenureYears" in df.columns and "TenureMonths" not in df.columns:
        df["TenureMonths"] = pd.to_numeric(df["TenureYears"], errors="coerce").fillna(0) * 12
    required = ["EmployeeID","Department","JobLevel","Gender","TenureMonths","AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().map(
        {"yes":"Yes","y":"Yes","1":"Yes","true":"Yes","no":"No","n":"No","0":"No","false":"No"}).fillna("No")

    total, left = len(df), (df["AttritionFlag"]=="Yes").sum()
    rate = (left/total*100) if total else 0
    avg_tenure = df["TenureMonths"].mean()

    c1,c2,c3 = st.columns(3)
    c1.metric("Attrition %", f"{rate:.1f}%")
    c2.metric("Avg Tenure (mo)", f"{avg_tenure:.1f}")
    c3.metric("Total Left", f"{left}")

    dept = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")
    job = df.groupby("JobLevel", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")
    df["TenureCohort"] = pd.cut(df["TenureMonths"], [-1,12,36,60,120], labels=["<1 yr","1–3 yrs","3–5 yrs","5+ yrs"])
    cohort = df.groupby("TenureCohort", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")

    fig_dept = px.bar(dept, x="Department", y="Rate", text="Rate", title="Attrition % by Department", color="Department", template="plotly_white")
    fig_job = px.bar(job, x="JobLevel", y="Rate", text="Rate", title="Attrition % by Job Level", color="JobLevel", template="plotly_white")
    fig_cohort = px.bar(cohort, x="TenureCohort", y="Rate", text="Rate", title="Attrition % by Tenure Cohort", color="TenureCohort", template="plotly_white")
    fig_reason = None
    reason_path = None
    if "ExitReason" in df.columns and df["ExitReason"].notna().any():
        reasons = df[df["AttritionFlag"]=="Yes"]["ExitReason"].value_counts().reset_index()
        reasons.columns = ["ExitReason","Count"]
        fig_reason = px.pie(reasons, names="ExitReason", values="Count", title="Top Exit Reasons", template="plotly_white")

    dept_path = save_chart_image("Attrition by Department", fig_dept)
    job_path = save_chart_image("Attrition by JobLevel", fig_job)
    cohort_path = save_chart_image("Attrition by Tenure", fig_cohort)
    if fig_reason:
        reason_path = save_chart_image("Exit Reasons", fig_reason)

    data_blocks = [
        {"title":"Departmental Attrition","desc":"Attrition % by department","df":dept,"fig_path":dept_path,
         "insights":[f"Highest attrition dept: {dept.iloc[0]['Department'] if not dept.empty else 'N/A'}"]},
        {"title":"Tenure Cohort Attrition","desc":"Attrition by tenure","df":cohort,"fig_path":cohort_path,
         "insights":[f"Overall attrition: {rate:.1f}%"]},
        {"title":"Job Level Attrition","desc":"Attrition by job level","df":job,"fig_path":job_path,"insights":[]}
    ]
    if reason_path:
        data_blocks.append({"title":"Exit Reasons","desc":"Top exit drivers","df":None,"fig_path":reason_path,"insights":[]})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Attrition Analytics Executive Report", "Attrition", data_blocks, "Attrition")