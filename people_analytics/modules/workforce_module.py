# ============================================
# modules/workforce_module.py — v3.0.1 | Executive (Dual-theme safe + sync save)
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image
from utils.fix_helper import ensure_chart_saved, safe_categorical

def run_workforce_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0B5E3D,#10B981);color:white;">
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
    female_pct = (df["Gender"].astype(str).str.lower()=="female").mean()*100
    job_levels = df["JobLevel"].nunique()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Employees", f"{total}")
    c2.metric("Female %", f"{female_pct:.1f}%")
    c3.metric("Job Levels", f"{job_levels}")

    # Headcount by level (horizontal bar)
    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount").sort_values("Headcount", ascending=True)
    fig_hc = px.bar(hc, x="Headcount", y="JobLevel", orientation="h", text="Headcount", title="Headcount by Job Level", color="JobLevel")
    fig_hc.update_layout(template="plotly_white")
    fig_hc.update_traces(marker_line_color='black', marker_line_width=1, textposition="outside")

    st.subheader("Headcount by Job Level")
    st.dataframe(hc, use_container_width=True)
    st.plotly_chart(fig_hc, use_container_width=True)

    # Manager spans (if available) - show summary only for exec report
    span_df = None
    fig_span = None
    if "ManagerID" in df.columns:
        manager_counts = df["ManagerID"].value_counts().reset_index()
        manager_counts.columns = ["ManagerID","DirectReports"]
        span_df = manager_counts
        fig_span = px.histogram(manager_counts, x="DirectReports", nbins=15, title="Distribution of Direct Reports per Manager")
        fig_span.update_layout(template="plotly_white")
        fig_span.update_traces(marker_line_color='black', marker_line_width=1)
        st.subheader("Manager Spans")
        # For app show aggregated summary, not raw long table
        st.metric("Avg span", f"{manager_counts['DirectReports'].mean():.1f}")
        st.plotly_chart(fig_span, use_container_width=True)

    # Skills tokenization (if available)
    skills_df = None
    fig_skills = None
    if "Skills" in df.columns:
        tokens = Counter()
        for v in df["Skills"].dropna().astype(str):
            parts = [x.strip().lower() for x in v.replace("|",",").split(",") if x.strip()]
            tokens.update(parts)
        if tokens:
            skills_df = pd.DataFrame(tokens.most_common(20), columns=["Skill","Count"])
            fig_skills = px.bar(skills_df.sort_values("Count", ascending=True), x="Count", y="Skill", orientation="h", title="Top 20 Skills")
            fig_skills.update_layout(template="plotly_white")
            fig_skills.update_traces(marker_line_color='black', marker_line_width=1)
            st.subheader("Top Skills")
            st.dataframe(skills_df, use_container_width=True)
            st.plotly_chart(fig_skills, use_container_width=True)

    # Prepare data blocks for PDF (exec summary style)
    data_blocks = [
        {"title":"Headcount Structure","desc":"Headcount distribution across job levels","df":hc,"fig":fig_hc,
         "insights":[f"Total employees: {total}", f"Female %: {female_pct:.1f}%"]},
    ]
    if span_df is not None:
        # keep manager spans summary table (top n managers) rather than full raw list
        top_spans = span_df.sort_values("DirectReports", ascending=False).head(10).reset_index(drop=True)
        data_blocks.append({"title":"Manager Spans","desc":"Top manager spans (top 10)","df":top_spans,"fig":fig_span,"insights":[f"Average span: {span_df['DirectReports'].mean():.1f}"]})
    if skills_df is not None:
        data_blocks.append({"title":"Skill Inventory","desc":"Top skills across the workforce","df":skills_df,"fig":fig_skills,"insights":[]})

    # Save charts synchronously
    saved_assets = {}
    for title, fig in [
        ("Headcount by Job Level", fig_hc),
        ("Manager Spans", fig_span),
        ("Top Skills", fig_skills)
    ]:
        if fig is not None:
            path = ensure_chart_saved(fig, title, save_chart_image)
            if path:
                saved_assets[title] = {"png": {"path": path}}

    # Attach assets into blocks (optional)
    for b in data_blocks:
        title = b.get("title")
        if saved_assets.get(title):
            b["asset"] = saved_assets.get(title)

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    if not data_blocks:
        st.warning("No data available to export.")
        return
    render_pdf_download_button("Workforce Analytics Executive Report","Workforce",data_blocks,"Workforce")
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
# 🔹 Auto-write metadata JSON for consolidated summary
import json
meta = {
    "insights": f"Total Employees {total:,} • Female {female_pct:.1f}% • Job Levels {job_levels}",
    "metrics_short": "Headcount, Gender %, Job Levels"
}
with open(os.path.join(TMP_DIR, "Workforce.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f)
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