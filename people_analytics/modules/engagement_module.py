# ============================================
# modules/engagement_module.py — v3.0.1 | Executive (Dual-theme safe + sync save)
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image
from utils.fix_helper import ensure_chart_saved

MODULE_COLOR = "#3B82F6"

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float","int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def run_engagement_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#1E40AF,#3B82F6);color:white;">
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

    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Engagement Index", f"{avg_index:.2f}")
    c2.metric("Highly Engaged %", f"{pct_high:.1f}%")
    c3.metric("Low Engaged %", f"{pct_low:.1f}%")

    # department summary
    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanIndex","MedianIndex","Count","StdDev"]
    dept_summary = _round_df(dept_summary)

    gender_summary = df.groupby("Gender", observed=True)["EngagementIndex"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanIndex","Count"]
    gender_summary = _round_df(gender_summary)

    # categories
    bins = [-1, 2.9, 3.6, 5]
    labels = ["Low","Medium","High"]
    df["EngagementCat"] = pd.cut(df["EngagementIndex"], bins=bins, labels=labels).astype(str)
    cat_counts = df["EngagementCat"].value_counts().reindex(labels, fill_value=0).reset_index()
    cat_counts.columns = ["Category","Count"]

    # visuals (white template to preserve colors in PDF)
    fig_dept = px.bar(dept_summary.sort_values("MeanIndex", ascending=False), x="Department", y="MeanIndex",
                      text="MeanIndex", title="Avg Engagement by Department")
    fig_dept.update_layout(template="plotly_white")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    fig_cat = px.pie(cat_counts, names="Category", values="Count", title="Engagement Categories")
    fig_cat.update_layout(template="plotly_white")

    fig_gender = px.bar(gender_summary, x="Gender", y="MeanIndex", text="MeanIndex", title="Avg Engagement by Gender")
    fig_gender.update_layout(template="plotly_white")
    fig_gender.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    # App display
    st.subheader("Department Engagement Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(fig_dept, use_container_width=True)

    st.subheader("Engagement Categories")
    st.dataframe(cat_counts, use_container_width=True)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Demographic Split")
    st.dataframe(gender_summary, use_container_width=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    # Save charts synchronously
    saved_assets = {}
    for title, fig in [
        ("Avg Engagement by Dept", fig_dept),
        ("Engagement Categories", fig_cat),
        ("Engagement by Gender", fig_gender)
    ]:
        if fig is not None:
            path = ensure_chart_saved(fig, title, save_chart_image)
            if path:
                saved_assets[title] = {"png": {"path": path}}

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
            "insights": [f"Top department: {dept_summary.sort_values('MeanIndex', ascending=False).iloc[0]['Department'] if not dept_summary.empty else 'N/A' }"],
            "asset": saved_assets.get("Avg Engagement by Dept")
        },
        {
            "title": "Engagement Categories",
            "desc": "High/Medium/Low segmentation of engagement.",
            "df": cat_counts,
            "fig": fig_cat,
            "insights": [f"High engagement share: {pct_high:.1f}%"],
            "asset": saved_assets.get("Engagement Categories")
        },
        {
            "title": "Demographic Engagement",
            "desc": "Engagement by gender.",
            "df": gender_summary,
            "fig": fig_gender,
            "insights": [],
            "asset": saved_assets.get("Engagement by Gender")
        }
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Engagement Analytics Executive Report", "Engagement", data_blocks, "Engagement")
# ============================================
# ➕ Add to Consolidated Leadership Deck (Unified)
# ============================================
import os, shutil
from utils_consolidated.pdf_merger import TMP_DIR
from utils_consolidated.deck_state_tracker import update_module_state

st.markdown("---")
st.subheader("🧩 Add to Consolidated Leadership Deck")

# Derive module name dynamically from file (e.g., "Workforce", "Compensation")
module_name = __name__.split("_")[0].replace("modules.", "").capitalize()
pdf_filename = f"{module_name}_Analytics_Executive_Report.pdf"

possible_paths = [
    os.path.join("/tmp", pdf_filename),
    os.path.join(os.getcwd(), pdf_filename)
]

existing_pdf = next((p for p in possible_paths if os.path.exists(p)), None)
dest_path = os.path.join(TMP_DIR, f"{module_name}.pdf")

# --- Check if already added ---
if os.path.exists(dest_path):
    st.success("✅ A copy of this report has been added to the consolidated deck queue.")
else:
    if existing_pdf:
        if st.button(f"➕ Add {module_name} Report to Consolidated Deck", use_container_width=True):
            try:
                shutil.copyfile(existing_pdf, dest_path)
                update_module_state(module_name)
                st.success("✅ A copy of this report has been added to the consolidated deck queue.")
            except Exception as e:
                st.error(f"⚠️ Failed to add report: {e}")
    else:
        st.info("⚙️ Generate the PDF first before adding to the consolidated deck.")