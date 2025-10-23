import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image

MODULE_COLOR = "#3B82F6"

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float","int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def run_engagement_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;
         background:linear-gradient(90deg,#1E40AF,#3B82F6);color:white;">
      <h2 style="margin:0">💬 Engagement Analytics</h2>
      <p style="margin:4px 0 0 0;">Survey index, departmental splits & category segmentation (Executive view).</p>
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

    avg_index = float(df["EngagementIndex"].mean())
    pct_high = float((df["EngagementIndex"] > 3.6).mean() * 100)
    pct_low = float((df["EngagementIndex"] <= 2.9).mean() * 100)
    response_count = int(len(df))

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Engagement Index", f"{avg_index:.2f}")
    c2.metric("Highly Engaged %", f"{pct_high:.1f}%")
    c3.metric("Low Engaged %", f"{pct_low:.1f}%")

    dept_summary = _round_df(df.groupby("Department", observed=True)["EngagementIndex"]
                             .agg(["mean","median","count","std"])
                             .reset_index()
                             .rename(columns={"mean":"MeanIndex","median":"MedianIndex",
                                              "count":"Count","std":"StdDev"}))
    gender_summary = _round_df(df.groupby("Gender", observed=True)["EngagementIndex"]
                               .agg(["mean","count"]).reset_index()
                               .rename(columns={"mean":"MeanIndex","count":"Count"}))

    bins = [-1,2.9,3.6,5]
    labels = ["Low","Medium","High"]
    df["EngagementCat"] = pd.cut(df["EngagementIndex"], bins=bins, labels=labels).astype(str)
    cat_counts = df["EngagementCat"].value_counts().reindex(labels, fill_value=0).reset_index()
    cat_counts.columns = ["Category","Count"]

    fig_dept = px.bar(dept_summary.sort_values("MeanIndex", ascending=False),
                      x="Department", y="MeanIndex", text="MeanIndex",
                      title="Avg Engagement by Department", template="plotly_white")
    fig_cat = px.pie(cat_counts, names="Category", values="Count",
                     title="Engagement Categories", template="plotly_white")
    fig_gender = px.bar(gender_summary, x="Gender", y="MeanIndex", text="MeanIndex",
                        title="Avg Engagement by Gender", template="plotly_white")

    dept_path = save_chart_image("Engagement by Department", fig_dept)
    cat_path = save_chart_image("Engagement Categories", fig_cat)
    gender_path = save_chart_image("Engagement by Gender", fig_gender)

    st.subheader("Department Engagement Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(fig_dept, use_container_width=True)
    st.subheader("Engagement Categories")
    st.dataframe(cat_counts, use_container_width=True)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.subheader("Demographic Split")
    st.dataframe(gender_summary, use_container_width=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    data_blocks = [
        {"title":"Engagement Index Overview","desc":"Overall engagement index (mean of Q* responses).",
         "df":pd.DataFrame([{"Metric":"AvgIndex","Value":round(avg_index,2)},
                            {"Metric":"Responses","Value":response_count}]),
         "fig_path":None,
         "insights":[f"Average engagement index: {avg_index:.2f}",
                     f"Highly engaged: {pct_high:.1f}%"]},
        {"title":"Departmental Engagement","desc":"Average engagement score by department.",
         "df":dept_summary,"fig_path":dept_path,
         "insights":[f"Top department: {dept_summary.sort_values('MeanIndex',ascending=False).iloc[0]['Department'] if not dept_summary.empty else 'N/A'}"]},
        {"title":"Engagement Categories","desc":"High/Medium/Low segmentation of engagement.",
         "df":cat_counts,"fig_path":cat_path,
         "insights":[f"High engagement share: {pct_high:.1f}%"]},
        {"title":"Demographic Engagement","desc":"Engagement by gender.",
         "df":gender_summary,"fig_path":gender_path,"insights":[]}
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Engagement Analytics Executive Report",
                               "Engagement", data_blocks, "Engagement")