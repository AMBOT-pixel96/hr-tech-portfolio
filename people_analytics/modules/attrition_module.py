# ============================================
# modules/attrition_module.py — v3.2 | Consolidated Deck + Timestamp Integration
# ============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import ensure_chart_saved

# ============================================
# Local helper (replaces removed util)
# ============================================
def safe_categorical(df, col):
    """Safely converts categorical columns to string to prevent grouping crashes."""
    if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str)
    return df


# ============================================
# Main Attrition Module
# ============================================
def run_attrition_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#7F1D1D,#DC2626);color:white;">
      <h2 style="margin:0">📉 Attrition Analytics</h2>
      <p style="margin:4px 0 0 0;">Turnover, tenure cohorts & exit reasons (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Attrition Data (CSV/XLSX)")
    if df is None:
        return

    # Required column validation
    required = ["EmployeeID", "Department", "JobLevel", "Gender", "TenureMonths", "AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    # Normalize AttritionFlag
    df["AttritionFlag"] = (
        df["AttritionFlag"].astype(str).str.strip().str.lower().map(
            {"yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes",
             "no": "No", "n": "No", "0": "No", "false": "No"}
        ).fillna("No")
    )

    # Key metrics
    total = len(df)
    left = (df["AttritionFlag"] == "Yes").sum()
    rate = (left / total * 100) if total else 0
    avg_tenure = df["TenureMonths"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Attrition %", f"{rate:.1f}%")
    c2.metric("Avg Tenure (mo)", f"{avg_tenure:.1f}")
    c3.metric("Total Left", f"{left}")

    # Departmental attrition
    dept = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="Rate")
    dept = dept.sort_values("Rate", ascending=False)

    # Job level attrition
    job = df.groupby("JobLevel", observed=True)["AttritionFlag"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="Rate")

    # Tenure cohort attrition
    df["TenureCohort"] = pd.cut(
        df["TenureMonths"],
        bins=[-1, 12, 36, 60, 120],
        labels=["<1 yr", "1–3 yrs", "3–5 yrs", "5+ yrs"]
    )
    df = safe_categorical(df, "TenureCohort")
    cohort = df.groupby("TenureCohort", observed=True)["AttritionFlag"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="Rate")

    # Exit reasons (optional)
    fig_reason = None
    if "ExitReason" in df.columns and df["ExitReason"].notna().any():
        reasons = df[df["AttritionFlag"] == "Yes"]["ExitReason"].value_counts().reset_index()
        reasons.columns = ["ExitReason", "Count"]
        fig_reason = px.pie(
            reasons,
            names="ExitReason",
            values="Count",
            title="Top Exit Reasons",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

    # ============================================
    # 🖼️ Charts — Color-safe and PDF-ready
    # ============================================
    fig_dept = px.bar(
        dept, x="Department", y="Rate", text="Rate", color="Department",
        title="Attrition % by Department", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_job = px.bar(
        job, x="JobLevel", y="Rate", text="Rate", color="JobLevel",
        title="Attrition % by Job Level", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_cohort = px.bar(
        cohort, x="TenureCohort", y="Rate", text="Rate", color="TenureCohort",
        title="Attrition % by Tenure Cohort", color_discrete_sequence=px.colors.qualitative.Vivid
    )

    for f in (fig_dept, fig_job, fig_cohort):
        f.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_color="black",
            marker_line_width=1
        )
        f.update_layout(template="plotly_white", font=dict(color="black"))

    # ============================================
    # 📊 Streamlit Display
    # ============================================
    st.subheader("Departmental Attrition")
    st.dataframe(dept, use_container_width=True)
    st.plotly_chart(fig_dept, use_container_width=True)

    st.subheader("Tenure Cohort Attrition")
    st.dataframe(cohort, use_container_width=True)
    st.plotly_chart(fig_cohort, use_container_width=True)

    st.subheader("Job Level Attrition")
    st.dataframe(job, use_container_width=True)
    st.plotly_chart(fig_job, use_container_width=True)

    if fig_reason:
        st.subheader("Exit Reasons")
        st.plotly_chart(fig_reason, use_container_width=True)

    # ============================================
    # 📄 PDF Export
    # ============================================
    data_blocks = [
        {
            "title": "Departmental Attrition",
            "desc": "Attrition % by department",
            "df": dept,
            "fig": fig_dept,
            "insights": [f"Highest attrition department: {dept.iloc[0]['Department'] if not dept.empty else 'N/A'}"]
        },
        {
            "title": "Tenure Cohort Attrition",
            "desc": "Attrition by tenure cohorts",
            "df": cohort,
            "fig": fig_cohort,
            "insights": [f"Overall attrition: {rate:.1f}%"]
        },
        {
            "title": "Job Level Attrition",
            "desc": "Attrition % by job level",
            "df": job,
            "fig": fig_job,
            "insights": []
        }
    ]
    if fig_reason:
        data_blocks.append({
            "title": "Exit Reasons",
            "desc": "Top exit drivers",
            "df": None,
            "fig": fig_reason,
            "insights": []
        })

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Attrition Analytics Executive Report", "Attrition", data_blocks, "Attrition")
# ============================================
# ➕ Add to Consolidated Leadership Deck (Attrition)
# ============================================
import os, shutil, json
from utils_consolidated.pdf_merger import TMP_DIR
from utils_consolidated.deck_state_tracker import update_module_state

st.markdown("---")
st.subheader("🧩 Add to Consolidated Leadership Deck")

module_name = "Attrition"
pdf_filename = f"{module_name}_Analytics_Executive_Report.pdf"

possible_paths = [
    os.path.join("/tmp", pdf_filename),
    os.path.join(os.getcwd(), pdf_filename)
]

existing_pdf = next((p for p in possible_paths if os.path.exists(p)), None)
dest_path = os.path.join(TMP_DIR, f"{module_name}.pdf")

# --- If already added ---
if os.path.exists(dest_path):
    st.success("✅ A copy of this report has been added to the consolidated deck queue.")

else:
    if existing_pdf:
        if st.button(f"➕ Add {module_name} Report to Consolidated Deck", use_container_width=True):
            try:
                shutil.copyfile(existing_pdf, dest_path)
                update_module_state(module_name)
                st.success("✅ A copy of this report has been added to the consolidated deck queue.")

                # 🔹 Auto-write metadata JSON for consolidated summary
                meta = {
                    "insights": f"Attrition {rate:.1f}% • Avg Tenure {avg_tenure:.1f} mo • Top Dept {dept.iloc[0]['Department'] if not dept.empty else 'N/A'}",
                    "metrics_short": "Attrition %, Avg Tenure, Top Department"
                }
                with open(os.path.join(TMP_DIR, "Attrition.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)

            except Exception as e:
                st.error(f"⚠️ Failed to add report: {e}")
    else:
        st.info("⚙️ Generate the PDF first before adding to the consolidated deck.")