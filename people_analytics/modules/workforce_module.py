import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image

def run_workforce_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;
         background:linear-gradient(90deg,#0B5E3D,#10B981);color:white;">
      <h2 style="margin:0">🏢 Workforce & Talent Analytics</h2>
      <p style="margin:4px 0 0 0;">Headcount, spans & skills (Executive view).</p>
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
    female_pct = (df["Gender"].astype(str).str.lower() == "female").mean() * 100
    job_levels = df["JobLevel"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", f"{total}")
    c2.metric("Female %", f"{female_pct:.1f}%")
    c3.metric("Job Levels", f"{job_levels}")

    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount").sort_values("Headcount", ascending=True)
    fig_hc = px.bar(hc, x="Headcount", y="JobLevel", orientation="h",
                    text="Headcount", title="Headcount by Job Level",
                    color="JobLevel", template="plotly_white")

    span_df = None
    fig_span = None
    if "ManagerID" in df.columns:
        manager_counts = df["ManagerID"].value_counts().reset_index()
        manager_counts.columns = ["ManagerID", "DirectReports"]
        span_df = manager_counts
        fig_span = px.histogram(manager_counts, x="DirectReports", nbins=15,
                                title="Distribution of Direct Reports per Manager",
                                template="plotly_white")

    skills_df = None
    fig_skills = None
    if "Skills" in df.columns:
        tokens = Counter()
        for v in df["Skills"].dropna().astype(str):
            parts = [x.strip().lower() for x in v.replace("|", ",").split(",") if x.strip()]
            tokens.update(parts)
        if tokens:
            skills_df = pd.DataFrame(tokens.most_common(20), columns=["Skill", "Count"])
            fig_skills = px.bar(skills_df.sort_values("Count", ascending=True),
                                x="Count", y="Skill", orientation="h",
                                title="Top 20 Skills", template="plotly_white")

    # Save images
    hc_path = save_chart_image("Headcount by Job Level", fig_hc)
    span_path = save_chart_image("Manager Spans", fig_span) if fig_span else None
    skills_path = save_chart_image("Top Skills", fig_skills) if fig_skills else None

    st.subheader("Headcount by Job Level")
    st.dataframe(hc, use_container_width=True)
    st.plotly_chart(fig_hc, use_container_width=True)

    if span_df is not None:
        st.subheader("Manager Spans")
        st.dataframe(span_df, use_container_width=True)
        st.plotly_chart(fig_span, use_container_width=True)
        st.caption(f"Average span: {span_df['DirectReports'].mean():.1f}")

    if skills_df is not None:
        st.subheader("Top Skills")
        st.dataframe(skills_df, use_container_width=True)
        st.plotly_chart(fig_skills, use_container_width=True)

    data_blocks = [
        {"title": "Headcount Structure", "desc": "Headcount distribution across job levels",
         "df": hc, "fig_path": hc_path,
         "insights": [f"Total employees: {total}", f"Female %: {female_pct:.1f}%"]},
    ]
    if span_df is not None:
        data_blocks.append({"title": "Manager Spans", "desc": "Direct reports per manager",
                            "df": span_df, "fig_path": span_path,
                            "insights": [f"Average span: {span_df['DirectReports'].mean():.1f}"]})
    if skills_df is not None:
        data_blocks.append({"title": "Skill Inventory", "desc": "Top skills across the workforce",
                            "df": skills_df, "fig_path": skills_path, "insights": []})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button