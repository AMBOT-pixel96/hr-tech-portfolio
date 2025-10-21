# modules/attrition_module.py — v2.8 | Executive Edition
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_attrition_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#7F1D1D,#DC2626);color:white;">
      <h2 style="margin:0">📉 Attrition Analytics</h2>
      <p style="margin:4px 0 0 0;">Turnover, tenure cohorts & exit drivers (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Attrition Data (CSV/XLSX)")
    if df is None:
        return

    # Tenure fix
    if "TenureMonths" not in df.columns and "TenureYears" in df.columns:
        df["TenureMonths"] = df["TenureYears"] * 12

    required = ["EmployeeID","Department","JobLevel","Gender","TenureMonths","AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.lower().map(
        {"yes":"Yes","y":"Yes","1":"Yes","true":"Yes","no":"No","n":"No","0":"No","false":"No"}).fillna("No")

    total = len(df)
    left = (df["AttritionFlag"]=="Yes").sum()
    rate = (left/total*100) if total else 0
    avg_tenure = df["TenureMonths"].mean()

    c1,c2,c3 = st.columns(3)
    c1.metric("Attrition %", f"{rate:.1f}%")
    c2.metric("Avg Tenure (mo)", f"{avg_tenure:.1f}")
    c3.metric("Total Left", f"{left}")

    # Group metrics
    dept = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")
    job = df.groupby("JobLevel", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")
    df["TenureCohort"] = pd.cut(df["TenureMonths"], bins=[-1,12,36,60,120], labels=["<1 yr","1–3 yrs","3–5 yrs","5+ yrs"]).astype(str)
    cohort = df.groupby("TenureCohort", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")

    fig_dept = px.bar(dept, x="Department", y="Rate", text="Rate", title="Attrition by Department", color="Department")
    fig_job = px.bar(job, x="JobLevel", y="Rate", text="Rate", title="Attrition by Job Level", color="JobLevel")
    fig_tenure = px.bar(cohort, x="TenureCohort", y="Rate", text="Rate", title="Attrition by Tenure Cohort", color="TenureCohort")
    for f in [fig_dept,fig_job,fig_tenure]:
        f.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color='black', marker_line_width=1)

    # Exit reason pie
    fig_reason = None
    if "ExitReason" in df.columns and df["ExitReason"].dropna().any():
        reasons = df[df["AttritionFlag"]=="Yes"]["ExitReason"].value_counts().reset_index()
        reasons.columns = ["ExitReason","Count"]
        fig_reason = px.pie(reasons, names="ExitReason", values="Count", title="Exit Reasons")

    data_blocks = [
        {"title":"Departmental Attrition","desc":"Attrition % by department","df":dept,"fig":fig_dept,
         "insights":[f"Highest attrition dept: {dept.sort_values('Rate',ascending=False).iloc[0]['Department']}"]},
        {"title":"Job Level Attrition","desc":"Attrition by job level","df":job,"fig":fig_job,"insights":[]},
        {"title":"Tenure Cohorts","desc":"Attrition by tenure","df":cohort,"fig":fig_tenure,"insights":[f"Overall attrition: {rate:.1f}%"]},
    ]
    if fig_reason is not None:
        data_blocks.append({"title":"Exit Reasons","desc":"Top reasons for leaving","df":None,"fig":fig_reason,"insights":[]})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Attrition Analytics Executive Report","Attrition",data_blocks,"Attrition")