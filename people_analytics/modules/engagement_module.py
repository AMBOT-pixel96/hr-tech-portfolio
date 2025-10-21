# modules/engagement_module.py — v2.8 | Executive Edition
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_engagement_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#1E3A8A,#3B82F6);color:white;">
      <h2 style="margin:0">💬 Engagement Analytics</h2>
      <p style="margin:4px 0 0 0;">Survey index, departmental splits & engagement categories (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Engagement Survey (CSV/XLSX)")
    if df is None:
        return

    # detect survey columns Q*
    qcols = [c for c in df.columns if str(c).strip().upper().startswith("Q")]
    if not qcols:
        st.error("No survey question columns found (expect columns starting with 'Q').")
        return

    # numeric conversion
    df[qcols] = df[qcols].apply(pd.to_numeric, errors="coerce")
    df["EngagementIndex"] = df[qcols].mean(axis=1)

    # KPI row
    avg_index = df["EngagementIndex"].mean()
    pct_high = (df["EngagementIndex"] > 3.6).mean() * 100  # threshold high
    pct_low = (df["EngagementIndex"] <= 2.9).mean() * 100
    response_count = len(df)
    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Engagement Index", f"{avg_index:.2f}")
    c2.metric("Highly Engaged %", f"{pct_high:.1f}%")
    c3.metric("Low Engaged %", f"{pct_low:.1f}%")

    # department and demographic summaries
    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanIndex","MedianIndex","Count","StdDev"]
    gender_summary = df.groupby("Gender", observed=True)["EngagementIndex"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanIndex","Count"]

    # categories
    bins = [-1, 2.9, 3.6, 5]
    labels = ["Low","Medium","High"]
    df["EngagementCat"] = pd.cut(df["EngagementIndex"], bins=bins, labels=labels).astype(str)
    cat_counts = df["EngagementCat"].value_counts().reindex(labels, fill_value=0).reset_index()
    cat_counts.columns = ["Category","Count"]

    # visuals
    fig_dept = px.bar(dept_summary.sort_values("MeanIndex", ascending=False), x="Department", y="MeanIndex", text="MeanIndex", title="Avg Engagement by Department")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color='black', marker_line_width=1)
    fig_cat = px.pie(cat_counts, names="Category", values="Count", title="Engagement Categories")
    fig_gender = px.bar(gender_summary, x="Gender", y="MeanIndex", text="MeanIndex", title="Avg Engagement by Gender")
    fig_gender.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    # page layout
    st.subheader("Department Engagement Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(fig_dept, use_container_width=True)

    st.subheader("Engagement Categories")
    st.dataframe(cat_counts, use_container_width=True)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Demographic Split")
    st.dataframe(gender_summary, use_container_width=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    # Prepare PDF blocks
    data_blocks = [
        {
            "title": "Engagement Index Overview",
            "desc": "Overall engagement index (mean of Q* responses).",
            "df": pd.DataFrame([{"Metric":"AvgIndex","Value":round(avg_index,2)}, {"Metric":"Responses","Value":response_count}]),
            "fig": None,
            "insights": [f"Average engagement index: {avg_index:.2f}", f"Highly engaged: {pct_high:.1f}%"]
        },
        {
            "title": "Departmental Engagement",
            "desc": "Average engagement score by department.",
            "df": dept_summary,
            "fig": fig_dept,
            "insights": [f"Top department: {dept_summary.sort_values('MeanIndex', ascending=False).iloc[0]['Department']}"]
        },
        {
            "title": "Engagement Categories",
            "desc": "High/Medium/Low segmentation of engagement.",
            "df": cat_counts,
            "fig": fig_cat,
            "insights": [f"High engagement share: {pct_high:.1f}%"]
        },
        {
            "title": "Demographic Engagement",
            "desc": "Engagement by gender.",
            "df": gender_summary,
            "fig": fig_gender,
            "insights": []
        }
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Engagement Analytics Executive Report", "Engagement", data_blocks, "Engagement")