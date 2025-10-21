# modules/engagement_module.py — v2.6
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_engagement_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#1E3A8A,#3B82F6);color:white;">
      <h2 style="margin:0">💬 Engagement Analytics</h2>
      <p style="margin:4px 0 0 0;">Survey index, departmental splits & engagement categories.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Engagement Survey (CSV/XLSX)")
    if df is None:
        return

    qcols = [c for c in df.columns if str(c).strip().upper().startswith("Q")]
    if not qcols:
        st.error("No survey question columns found (expect columns starting with 'Q').")
        return

    df[qcols] = df[qcols].apply(pd.to_numeric, errors="coerce")
    df["EngagementIndex"] = df[qcols].mean(axis=1)

    avg_index = round(df["EngagementIndex"].mean(),2)
    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index().sort_values("EngagementIndex", ascending=False)

    # Bin into High/Medium/Low (default thresholds — you can change)
    bins = [ -1, 2.9, 3.6, 5 ]  # Low <=2.9, Medium 3.0-3.6, High >3.6
    labels = ["Low","Medium","High"]
    df["EngagementCat"] = pd.cut(df["EngagementIndex"], bins=bins, labels=labels).astype(str)
    cat_counts = df["EngagementCat"].value_counts().reset_index()
    cat_counts.columns = ["Category","Count"]

    # Visuals
    fig_dept = px.bar(dept_summary, x="Department", y="EngagementIndex", text="EngagementIndex", title="Avg Engagement by Department", color="Department")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_pie = px.pie(cat_counts, names="Category", values="Count", title="Engagement Categories")

    c1,c2 = st.columns(2)
    c1.metric("Avg Engagement Index", f"{avg_index}")
    c2.metric("Highly Engaged %", f"{round((df['EngagementCat']=='High').mean()*100,1)}%")

    st.plotly_chart(fig_dept, use_container_width=True)
    st.plotly_chart(fig_pie, use_container_width=True)

    # PDF blocks
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [
        {"title":"Engagement by Dept","desc":"Avg engagement per department","df":dept_summary,"fig":fig_dept,"insights":[f"Avg index: {avg_index}"]},
        {"title":"Engagement Categories","desc":"High / Medium / Low distribution","df":cat_counts,"fig":fig_pie,"insights":[]}
    ]
    render_pdf_download_button("Engagement Analytics Executive Report","Engagement",data_blocks,"Engagement")