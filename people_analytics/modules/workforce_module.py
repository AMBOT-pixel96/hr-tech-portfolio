# modules/workforce_module.py — v2.8 | Executive Edition
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_workforce_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0B5E3D,#10B981);color:white;">
      <h2 style="margin:0">🏢 Workforce & Talent Analytics</h2>
      <p style="margin:4px 0 0 0;">Structure, manager spans & skill inventory (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Workforce Data (CSV/XLSX)")
    if df is None:
        return

    required = ["EmployeeID","JobLevel","Gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    total = len(df)
    female_pct = (df["Gender"].str.lower()=="female").mean()*100
    job_levels = df["JobLevel"].nunique()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Employees", f"{total}")
    c2.metric("Female %", f"{female_pct:.1f}%")
    c3.metric("Job Levels", f"{job_levels}")

    # Headcount by level
    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount").sort_values("Headcount", ascending=True)
    fig_hc = px.bar(hc, x="Headcount", y="JobLevel", orientation="h", text="Headcount", title="Headcount by Job Level", color="JobLevel")
    fig_hc.update_traces(marker_line_color='black', marker_line_width=1, textposition="outside")

    # Manager Spans
    span_df = None
    fig_span = None
    if "ManagerID" in df.columns:
        manager_counts = df["ManagerID"].value_counts().reset_index()
        manager_counts.columns = ["ManagerID","DirectReports"]
        span_df = manager_counts
        fig_span = px.histogram(manager_counts, x="DirectReports", nbins=15, title="Distribution of Direct Reports per Manager")
        st.caption(f"Average span: {manager_counts['DirectReports'].mean():.1f}")

    # Skills
    fig_skills = None
    skills_df = None
    if "Skills" in df.columns:
        tokens = Counter([s.strip().lower() for val in df["Skills"].dropna() for s in val.replace("|",",").split(",") if s.strip()])
        if tokens:
            skills_df = pd.DataFrame(tokens.most_common(20), columns=["Skill","Count"])
            fig_skills = px.bar(skills_df.sort_values("Count", ascending=True), x="Count", y="Skill",
                                orientation="h", title="Top 20 Skills")
            fig_skills.update_traces(marker_line_color='black', marker_line_width=1)

    data_blocks = [
        {"title":"Headcount Structure","desc":"Headcount distribution across job levels","df":hc,"fig":fig_hc,"insights":[f"Total employees: {total}"]},
    ]
    if span_df is not None:
        data_blocks.append({"title":"Manager Spans","desc":"Direct reports per manager","df":span_df,"fig":fig_span,"insights":[]})
    if skills_df is not None:
        data_blocks.append({"title":"Skill Inventory","desc":"Top 20 skills by frequency","df":skills_df,"fig":fig_skills,"insights":[]})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Workforce Analytics Executive Report","Workforce",data_blocks,"Workforce")