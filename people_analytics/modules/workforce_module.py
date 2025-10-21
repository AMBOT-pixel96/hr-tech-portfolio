# modules/workforce_module.py — v2.6
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def _tokenize_skills(skills_series, sep_chars=[";","|",","]):
    """Tokenize a skills column (returns Counter)"""
    tokens = []
    for v in skills_series.dropna().astype(str):
        for sep in sep_chars:
            if sep in v:
                parts = [p.strip() for p in v.split(sep) if p.strip()]
                tokens.extend(parts)
                break
        else:
            # no separator found — treat as single skill or space separated
            parts = [p.strip() for p in v.split(",") if p.strip()]
            tokens.extend(parts)
    return Counter([t.lower() for t in tokens if t])

def run_workforce_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0B5E3D,#10B981);color:white;">
      <h2 style="margin:0">🏢 Workforce Analytics</h2>
      <p style="margin:4px 0 0 0;">Headcount, manager spans & skill inventory.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Workforce Data (CSV/XLSX)")
    if df is None:
        return

    required = ["EmployeeID","JobLevel","Gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Basic KPIs
    total = len(df)
    female_pct = round((df["Gender"].fillna("Unknown").str.lower()=="female").mean()*100,1)
    levels = df["JobLevel"].nunique()

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Employees", f"{total}")
    c2.metric("Female %", f"{female_pct}%")
    c3.metric("Unique Job Levels", f"{levels}")

    # Headcount by level
    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount").sort_values("Headcount", ascending=True)
    fig_hc = px.bar(hc, x="Headcount", y="JobLevel", orientation="h", text="Headcount", title="Headcount by Job Level")
    fig_hc.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig_hc, use_container_width=True)

    # Manager spans (if ManagerID present)
    span_df = None
    span_fig = None
    if "ManagerID" in df.columns:
        # count direct reports per manager
        manager_counts = df["ManagerID"].value_counts().reset_index()
        manager_counts.columns = ["ManagerID","DirectReports"]
        span_df = manager_counts
        span_fig = px.histogram(span_df, x="DirectReports", nbins=20, title="Distribution of Direct Reports per Manager")
        st.plotly_chart(span_fig, use_container_width=True)
        avg_span = round(manager_counts["DirectReports"].mean(),2)
        st.caption(f"Avg manager span: {avg_span}")
    else:
        st.info("ManagerID column not found — manager span chart skipped.")

    # Skills tokenization (if Skills present)
    skills_cnt = None
    skills_fig = None
    if "Skills" in df.columns:
        cnt = _tokenize_skills(df["Skills"])
        if cnt:
            skills_cnt = pd.DataFrame(cnt.most_common(20), columns=["Skill","Count"])
            skills_fig = px.bar(skills_cnt.sort_values("Count", ascending=True), x="Count", y="Skill", orientation="h", title="Top Skills (Top 20)")
            st.plotly_chart(skills_fig, use_container_width=True)
        else:
            st.info("Skills column present but no tokens found.")
    else:
        st.info("Skills column not found — skill inventory skipped.")

    # PDF blocks
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [
        {"title":"Headcount by Level","desc":"Headcount distribution","df":hc,"fig":fig_hc,"insights":[f"Total employees: {total}"]},
    ]
    if span_df is not None:
        data_blocks.append({"title":"Manager Spans","desc":"Direct reports per manager","df":span_df,"fig":span_fig,"insights":[]})
    if skills_cnt is not None:
        data_blocks.append({"title":"Skill Inventory","desc":"Top skills","df":skills_cnt,"fig":skills_fig,"insights":[]})

    render_pdf_download_button("Workforce Analytics Executive Report","Workforce",data_blocks,"Workforce")