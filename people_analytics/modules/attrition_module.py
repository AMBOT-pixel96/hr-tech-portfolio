# modules/attrition_module.py — v2.6
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_attrition_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#7F1D1D,#DC2626);color:white;">
      <h2 style="margin:0">📉 Attrition Analytics</h2>
      <p style="margin:4px 0 0 0;">Turnover, tenure cohorts & exit drivers.</p>
    </div>
    """, unsafe_allow_html=True)

    # sample template download
    sample = pd.DataFrame({
        "EmployeeID":["E001","E002"],
        "Department":["Finance","IT"],
        "JobLevel":["Analyst","Manager"],
        "Gender":["Male","Female"],
        "TenureYears":[2,5],
        "AttritionFlag":["Yes","No"],
        "ExitReason":["Better Pay",""]
    })
    render_download_template("Attrition Data Template", sample, "Attrition_Template.csv")

    df = upload_data("Upload Attrition Data (CSV/XLSX)")
    if df is None:
        return

    # Support TenureYears -> TenureMonths conversion from HR_DataForge
    if "TenureMonths" not in df.columns and "TenureYears" in df.columns:
        try:
            df["TenureMonths"] = (pd.to_numeric(df["TenureYears"], errors="coerce") * 12).round().fillna(0).astype(int)
        except Exception:
            df["TenureMonths"] = 0

    required = ["EmployeeID","Department","JobLevel","Gender","TenureMonths","AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    # Normalize AttritionFlag
    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().map({
        "yes":"Yes","y":"Yes","1":"Yes","true":"Yes","no":"No","n":"No","0":"No","false":"No"
    }).fillna("No")

    # Tenure cohorts and safe string cast (avoid categorical setitem)
    df["TenureMonths"] = pd.to_numeric(df["TenureMonths"], errors="coerce").fillna(0)
    df["TenureCohort"] = pd.cut(df["TenureMonths"], bins=[-1,12,36,60,120],
                                labels=["<1 yr","1–3 yrs","3–5 yrs","5+ yrs"], include_lowest=True)
    df["TenureCohort"] = df["TenureCohort"].astype(str)

    # KPIs
    total = len(df)
    left = int((df["AttritionFlag"]=="Yes").sum())
    rate = round(left/total*100,1) if total else 0.0
    avg_tenure = round(df["TenureMonths"].mean(),1) if total else 0.0

    c1,c2,c3 = st.columns(3)
    c1.metric("Overall Attrition %", f"{rate}%")
    c2.metric("Avg Tenure (months)", f"{avg_tenure}")
    c3.metric("Total Left", f"{left} of {total}")

    # Dept / Job / Tenure charts
    dept_summary = (df.groupby("Department", observed=True)["AttritionFlag"]
                      .apply(lambda x: (x=="Yes").mean()*100).reset_index(name="Rate"))
    job_summary = (df.groupby("JobLevel", observed=True)["AttritionFlag"]
                     .apply(lambda x: (x=="Yes").mean()*100).reset_index(name="Rate"))
    tenure_summary = (df.groupby("TenureCohort", observed=True)["AttritionFlag"]
                        .apply(lambda x: (x=="Yes").mean()*100).reset_index(name="Rate"))

    fig_dept = px.bar(dept_summary, x="Department", y="Rate", text="Rate", title="Attrition % by Department", color="Department")
    fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_job = px.bar(job_summary, x="JobLevel", y="Rate", text="Rate", title="Attrition % by Job Level", color="JobLevel")
    fig_job.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_tenure = px.bar(tenure_summary, x="TenureCohort", y="Rate", text="Rate", title="Attrition by Tenure Cohort", color="TenureCohort")
    fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    st.plotly_chart(fig_dept, use_container_width=True)
    st.plotly_chart(fig_job, use_container_width=True)
    st.plotly_chart(fig_tenure, use_container_width=True)

    # Exit reasons pie
    if "ExitReason" in df.columns and not df[df["AttritionFlag"]=="Yes"]["ExitReason"].dropna().empty:
        reasons = (df[df["AttritionFlag"]=="Yes"]["ExitReason"].value_counts().reset_index())
        reasons.columns = ["ExitReason","Count"]
        fig_reason = px.pie(reasons, names="ExitReason", values="Count", title="Top Exit Reasons")
        st.plotly_chart(fig_reason, use_container_width=True)
    else:
        st.info("No ExitReason data available for pie chart.")

    # Build PDF data_blocks (include figs)
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [
        {"title":"Department Attrition","desc":"Attrition rates by department","df":dept_summary,"fig":fig_dept,
         "insights":[f"Overall attrition: {rate}%",f"Avg tenure: {avg_tenure} months"]},
        {"title":"Job Level Attrition","desc":"Attrition by job level","df":job_summary,"fig":fig_job,"insights":[]},
        {"title":"Tenure Cohorts","desc":"Attrition by tenure band","df":tenure_summary,"fig":fig_tenure,"insights":[]}
    ]
    if "fig_reason" in locals():
        data_blocks.append({"title":"Exit Reasons","desc":"Why employees left","df":reasons,"fig":fig_reason,"insights":[]})

    render_pdf_download_button("Attrition Analytics Executive Report","Attrition",data_blocks,"Attrition")