# modules/workforce_module.py — v1.0 | Workforce & Talent Planning + PDF Export
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_workforce_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#0B5E3D,#10B981);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏢 Workforce & Talent Planning</h2>
        <p style="font-size:14px; margin-top:6px;">
            Headcount, manager spans, org pyramid and a lightweight skill inventory analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------
    # Template download
    # -----------------------
    st.subheader("📄 Step 1 — Download Workforce Template")
    sample = pd.DataFrame([
        {"EmployeeID": "E1001", "ManagerID": "M001", "Department": "Finance", "JobLevel": "Analyst",
         "JobRole": "Analyst", "Gender": "Male", "TenureMonths": 24, "CTC": 600000, "Skills": "Excel,PowerBI"}
    ])
    render_download_template("Workforce Data Template", sample, "Workforce_Template.csv")

    # -----------------------
    # Upload
    # -----------------------
    st.subheader("📤 Step 2 — Upload Workforce Dataset")
    wf_file = st.file_uploader(
        "Upload Workforce Data (CSV or Excel)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )
    if wf_file is None:
        st.info("Please upload workforce data (EmployeeID, ManagerID (optional), Department, JobLevel, Skills).")
        return

    try:
        if wf_file.name.endswith(".csv"):
            df = pd.read_csv(wf_file)
        else:
            df = pd.read_excel(wf_file, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # -----------------------
    # Validation
    # -----------------------
    required = ["EmployeeID", "Department", "JobLevel", "JobRole"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # -----------------------
    # Headcount / Pyramid
    # -----------------------
    st.subheader("📊 Headcount & Org Pyramid")
    headcount = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    headcount = headcount.sort_values(by="Headcount", ascending=False)
    st.dataframe(headcount, use_container_width=True)

    fig_pyramid = px.bar(headcount, x="Headcount", y="JobLevel", orientation="h", title="Headcount by Job Level", text="Headcount")
    st.plotly_chart(fig_pyramid, use_container_width=True)

    # -----------------------
    # Manager spans
    # -----------------------
    st.subheader("🧭 Manager Span Analysis")
    if "ManagerID" in df.columns:
        # managers who appear as ManagerID
        mgr_counts = df.groupby("ManagerID", observed=True)["EmployeeID"].nunique().reset_index(name="DirectReports")
        mgr_counts = mgr_counts.sort_values(by="DirectReports", ascending=False)
        avg_span = mgr_counts["DirectReports"].mean() if not mgr_counts.empty else 0
        st.metric("Average Span (direct reports per manager)", f"{avg_span:.2f}")
        st.dataframe(mgr_counts.head(50), use_container_width=True)
        fig_span = px.histogram(mgr_counts, x="DirectReports", nbins=20, title="Distribution of Manager Spans")
        st.plotly_chart(fig_span, use_container_width=True)
    else:
        st.info("ManagerID column not found. Add ManagerID to compute spans.")

    # -----------------------
    # Skill inventory (light)
    # -----------------------
    st.subheader("🧠 Skill Inventory (Light)")
    if "Skills" in df.columns:
        # Skills assumed comma-separated; normalize
        skills_series = (
            df["Skills"].fillna("")
              .astype(str)
              .str.split(",")
              .explode()
              .str.strip()
              .replace("", pd.NA)
              .dropna()
        )
        if not skills_series.empty:
            skills_count = skills_series.value_counts().reset_index()
            skills_count.columns = ["Skill", "Count"]
            st.dataframe(skills_count.head(30), use_container_width=True)
            fig_sk = px.bar(skills_count.head(20), x="Skill", y="Count", title="Top Skills (by count)", text="Count")
            fig_sk.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_sk, use_container_width=True)
        else:
            st.info("No skill tokens found after parsing 'Skills' column.")
    else:
        st.info("Skills column not found. You can add a comma-separated 'Skills' column for inventory analysis.")

    # -----------------------
    # Insights summary
    # -----------------------
    st.subheader("💡 Key Insights")
    top_level = headcount.iloc[0]["JobLevel"] if not headcount.empty else "N/A"
    insight_text = [
        f"Top Job Level by headcount: {top_level}",
    ]
    if "ManagerID" in df.columns and 'avg_span' in locals():
        insight_text.append(f"Average direct reports per manager: {avg_span:.2f}")
    if "Skills" in df.columns and not skills_series.empty:
        top_skill = skills_count.iloc[0]["Skill"]
        insight_text.append(f"Top skill: {top_skill}")

    st.markdown("<ul>" + "".join(f"<li>{x}</li>" for x in insight_text) + "</ul>", unsafe_allow_html=True)

# ==================================
# 📄 Export Executive Report
# ==================================
st.markdown("---")
st.subheader("📄 Step 5 — Export Executive Report")

data_blocks = [
    {
        "title": "Workforce Overview",
        "desc": "Headcount, structure, and span of control analytics summarized.",
        "df": df.head(10) if "df" in locals() else None,
        "insights": [
            "Organizational hierarchy and manager spans analyzed.",
            "Skill inventory and job-level distribution visualized."
        ],
    }
]

from utils.pdf_auto_exporter import export_module_report
export_module_report(
    report_title="Workforce Analytics Executive Report",
    module_name="Workforce & Talent",
    data_blocks=data_blocks,
    filename_prefix="Workforce"
)