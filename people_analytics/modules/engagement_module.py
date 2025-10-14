# modules/engagement_module.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from utils.template_helper import render_download_template

# --- Config: expected columns (demo + Q1..Qn) ---
DEFAULT_Q = 10
DEMOGRAPHIC_COLS = ["EmployeeID", "Department", "JobLevel", "Gender"]

def _detect_question_cols(df):
    return [c for c in df.columns if c.upper().startswith("Q") and c[1:].isdigit()]

def _validate_survey(df, min_questions=4):
    missing = [c for c in DEMOGRAPHIC_COLS if c not in df.columns]
    if missing:
        return False, f"Missing demographic columns: {', '.join(missing)}"
    qcols = _detect_question_cols(df)
    if len(qcols) < min_questions:
        return False, f"Not enough question columns found (need >= {min_questions}). Found: {len(qcols)}"
    return True, qcols

def run_engagement_module():
    st.header("💬 Employee Engagement")

    st.markdown(
        "Download the survey template, circulate it, and upload the filled responses. "
        "Answers must be integers 1 (Strongly Disagree) — 5 (Strongly Agree)."
    )

    # ---- Template download ----
    st.subheader("📥 Download Survey Template")
    num_q = st.number_input("Number of questions (template)", min_value=4, max_value=30, value=DEFAULT_Q, step=1)
    template_bytes, template_name = render_download_template(num_questions=int(num_q))
    st.download_button("Download Engagement Survey Template (Excel)", data=template_bytes, file_name=template_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")

    # ---- Upload filled survey ----
    st.subheader("📤 Upload Completed Survey")
    uploaded = st.file_uploader("Upload filled survey (Excel/CSV)", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Upload a completed survey to run the analysis.")
        return

    # Read file
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Unable to read file: {e}")
        return

    st.write("Preview (first 5 rows):")
    st.dataframe(df.head(), use_container_width=True)

    # Validate
    ok, result = _validate_survey(df, min_questions=4)
    if not ok:
        st.error(result)
        return
    qcols = result
    st.success(f"Detected {len(qcols)} question columns: {', '.join(qcols[:10])}{'...' if len(qcols)>10 else ''}")

    # Cast questions to numeric and clip 1-5
    for q in qcols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)

    # Drop rows with all-NaN answers
    df = df.dropna(subset=qcols, how="all")

    if df.empty:
        st.error("No valid response rows after processing question columns.")
        return

    # ---- Engagement Index (per respondent) ----
    df["EngagementIndex"] = df[qcols].mean(axis=1)  # 1-5 scale
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"],
        include_lowest=True
    )

    # Overall metrics
    overall_index = df["EngagementIndex"].mean()
    st.metric("Overall Engagement Index (1-5)", f"{overall_index:.2f}")

    # ---- Department-level averages ----
    st.subheader("📊 Engagement by Department")
    dept_avg = df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index().sort_values("EngagementIndex", ascending=False)
    dept_avg["EngagementIndex"] = dept_avg["EngagementIndex"].round(2)
    st.dataframe(dept_avg, use_container_width=True)

    bar = px.bar(dept_avg, x="Department", y="EngagementIndex", text="EngagementIndex", title="Average Engagement by Department")
    st.plotly_chart(bar, use_container_width=True)

    # ---- Question heatmap (which questions score high/low by dept) ----
    st.subheader("🔎 Question Heatmap by Department")
    # Pivot: dept x question average
    heat_df = df.groupby("Department")[qcols].mean().reset_index().set_index("Department")
    if heat_df.shape[0] == 0 or heat_df.shape[1] == 0:
        st.info("Not enough data to build heatmap.")
    else:
        # plotly heatmap expects numeric matrix
        fig = px.imshow(
            heat_df.values,
            x=heat_df.columns,
            y=heat_df.index,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            origin="lower",
            labels=dict(x="Question", y="Department", color="Avg Score"),
            title="Average Question Scores by Department (1-5)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Respondent distribution ----
    st.subheader("👥 Responses Breakdown")
    cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
    p = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Category Distribution")
    st.plotly_chart(p, use_container_width=True)

    # ---- Download aggregated results ----
    st.subheader("📤 Export Processed Data")
    to_export = df[[*DEMOGRAPHIC_COLS, *qcols, "EngagementIndex", "EngagementCategory"]]
    csv_bytes = to_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download Processed Responses (CSV)", csv_bytes, file_name="engagement_processed.csv", mime="text/csv")

    st.success("✅ Engagement analysis complete.")