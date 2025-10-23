# modules/consolidated_module.py — v1.0 | Consolidated HR Reporting Engine
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import ensure_chart_saved

# NOTE: these helpers are mostly lifted from your module logic but
# packaged so consolidated flow can produce data_blocks programmatically.

# ---------------------------
# 1) Performance helper
# ---------------------------
def get_performance_blocks(df: pd.DataFrame):
    if df is None or df.empty:
        return []

    # ensure numeric
    df["PerformanceRating"] = pd.to_numeric(df.get("PerformanceRating", pd.Series()), errors="coerce")
    df["CTC"] = pd.to_numeric(df.get("CTC", pd.Series()), errors="coerce")

    avg_rating = float(df["PerformanceRating"].mean())
    rating_std = float(df["PerformanceRating"].std())
    avg_ctc = float(df["CTC"].mean())
    top_perf_share = float((df["PerformanceRating"] >= 4).mean() * 100)
    low_perf_share = float((df["PerformanceRating"] <= 2).mean() * 100)

    dept_summary = df.groupby("Department", observed=True)["PerformanceRating"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanRating","MedianRating","Count","StdDev"]
    dept_summary = dept_summary.round(2)

    job_summary = df.groupby("JobLevel", observed=True)["PerformanceRating"].agg(["mean","median","count"]).reset_index()
    job_summary.columns = ["JobLevel","MeanRating","MedianRating","Count"]
    job_summary = job_summary.round(2)

    gender_summary = df.groupby("Gender", observed=True)["PerformanceRating"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanRating","Count"]
    gender_summary = gender_summary.round(2)

    # Figures
    box_dept = px.box(df, x="Department", y="PerformanceRating", title="Performance Ratings by Department", color="Department")
    box_dept.update_layout(template="plotly_white")

    box_ctc_by_rating = px.box(df, x="PerformanceRating", y="CTC", title="CTC distribution by Performance Rating", color="PerformanceRating")
    box_ctc_by_rating.update_layout(template="plotly_white")

    kde_fig = None
    x = df["PerformanceRating"].dropna()
    if len(x) > 3:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(x)
            x_range = np.linspace(max(x.min(), 0), x.max(), 200)
            y = kde(x_range)
            kde_df = pd.DataFrame({"Rating": x_range, "Density": y})
            kde_fig = px.line(kde_df, x="Rating", y="Density", title="Performance Rating Distribution (KDE)")
            kde_fig.update_layout(template="plotly_white")
        except Exception:
            kde_fig = None

    blocks = [
        {
            "title": "Performance Distribution",
            "desc": "Distribution summary: average, std, top/low shares.",
            "df": pd.DataFrame([{"Metric":"AvgRating","Value":round(avg_rating,2)},
                                {"Metric":"RatingStd","Value":round(rating_std,2)},
                                {"Metric":"AvgCTC","Value":f"₹{avg_ctc:,.0f}"}]),
            "fig": kde_fig,
            "insights":[f"Average rating: {avg_rating:.2f}",
                        f"Rating StdDev: {rating_std:.2f}",
                        f"Top performers (>=4): {top_perf_share:.1f}%",
                        f"Low performers (<=2): {low_perf_share:.1f}%"]
        },
        {"title":"Department Ratings","desc":"Mean and variation of ratings per department","df":dept_summary,"fig":box_dept,"insights":[]},
        {"title":"Performance vs Pay","desc":"CTC distribution across rating tiers","df":job_summary,"fig":box_ctc_by_rating,"insights":[f"Average CTC: ₹{avg_ctc:,.0f}"]},
        {"title":"Gender Performance","desc":"Mean ratings by gender.","df":gender_summary,"fig":None,"insights":[]}
    ]
    return blocks

# ---------------------------
# 2) Attrition helper
# ---------------------------
def get_attrition_blocks(df: pd.DataFrame):
    if df is None or df.empty:
        return []

    # Normalize
    if "TenureYears" in df.columns and "TenureMonths" not in df.columns:
        df["TenureMonths"] = pd.to_numeric(df["TenureYears"], errors="coerce").fillna(0) * 12

    required = ["EmployeeID","Department","JobLevel","Gender","TenureMonths","AttritionFlag"]
    for c in required:
        if c not in df.columns:
            # return an empty block indicating missing data
            return [{"title":"Attrition Data Missing","desc":f"Required column {c} not found","df":None,"fig":None,"insights":[]}]

    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().map(
        {"yes":"Yes","y":"Yes","1":"Yes","true":"Yes","no":"No","n":"No","0":"No","false":"No"}
    ).fillna("No")

    total = len(df)
    left = (df["AttritionFlag"]=="Yes").sum()
    rate = (left/total*100) if total else 0
    avg_tenure = df["TenureMonths"].mean()

    dept = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x: (x=="Yes").mean()*100).reset_index(name="Rate")
    job = df.groupby("JobLevel", observed=True)["AttritionFlag"].apply(lambda x: (x=="Yes").mean()*100).reset_index(name="Rate")

    df["TenureCohort"] = pd.cut(df["TenureMonths"], bins=[-1,12,36,60,120], labels=["<1 yr","1–3 yrs","3–5 yrs","5+ yrs"])
    cohort = df.groupby("TenureCohort", observed=True)["AttritionFlag"].apply(lambda x:(x=="Yes").mean()*100).reset_index(name="Rate")

    fig_dept = px.bar(dept, x="Department", y="Rate", text="Rate", title="Attrition % by Department", color="Department")
    fig_job = px.bar(job, x="JobLevel", y="Rate", text="Rate", title="Attrition % by Job Level", color="JobLevel")
    fig_cohort = px.bar(cohort, x="TenureCohort", y="Rate", text="Rate", title="Attrition % by Tenure Cohort", color="TenureCohort")

    fig_reason = None
    if "ExitReason" in df.columns and df["ExitReason"].notna().any():
        reasons = df[df["AttritionFlag"]=="Yes"]["ExitReason"].value_counts().reset_index()
        reasons.columns = ["ExitReason","Count"]
        if not reasons.empty:
            fig_reason = px.pie(reasons, names="ExitReason", values="Count", title="Top Exit Reasons")

    blocks = [
        {"title":"Attrition KPIs","desc":"Attrition % and tenure summary","df":pd.DataFrame([{"Metric":"Attrition%","Value":f"{rate:.1f}%"},
                                                                                     {"Metric":"AvgTenureMo","Value":f"{avg_tenure:.1f}"},
                                                                                     {"Metric":"TotalLeft","Value":left}]),"fig":None,"insights":[f"Overall attrition: {rate:.1f}%"]},
        {"title":"Departmental Attrition","desc":"Attrition % by department","df":dept,"fig":fig_dept,"insights":[]},
        {"title":"Tenure Cohort Attrition","desc":"Attrition by tenure cohort","df":cohort,"fig":fig_cohort,"insights":[]},
        {"title":"Job Level Attrition","desc":"Attrition by job level","df":job,"fig":fig_job,"insights":[]}
    ]
    if fig_reason:
        blocks.append({"title":"Exit Reasons","desc":"Top exit drivers","df":None,"fig":fig_reason,"insights":[]})

    return blocks

# ---------------------------
# 3) Compensation helper
# ---------------------------
def get_compensation_blocks(emp_df: pd.DataFrame, bench_df: pd.DataFrame = None):
    if emp_df is None or emp_df.empty:
        return []

    required = ["EmployeeID","Gender","Department","JobRole","JobLevel","CTC","Bonus"]
    for c in required:
        if c not in emp_df.columns:
            return [{"title":"Compensation Data Missing","desc":f"Required column {c} not found","df":None,"fig":None,"insights":[]}]

    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce").fillna(0)
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce").fillna(0)
    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"].replace({0:pd.NA})) * 100
    emp_df["BonusPct"] = emp_df["BonusPct"].fillna(0)

    avg_ctc = float(emp_df["CTC"].mean())
    avg_bonus_pct = float(emp_df["BonusPct"].mean())

    ctc_level = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index().rename(columns={"CTC":"AvgCTC"})
    bonus_level = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index().rename(columns={"BonusPct":"AvgBonusPct"})
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index().rename(columns={"CTC":"AvgCTC"})

    fig_ctc = px.bar(ctc_level, x="JobLevel", y="AvgCTC", text="AvgCTC", title="Avg CTC by Job Level", color="JobLevel")
    fig_ctc.update_layout(template="plotly_white")
    fig_bonus = px.bar(bonus_level, x="JobLevel", y="AvgBonusPct", text="AvgBonusPct", title="Avg Bonus % by Job Level", color="JobLevel")
    fig_bonus.update_layout(template="plotly_white")
    fig_gender = px.bar(gender_gap, x="Gender", y="AvgCTC", text="AvgCTC", title="Avg CTC by Gender", color="Gender")
    fig_gender.update_layout(template="plotly_white")

    comp_summary = None
    fig_market = None
    if bench_df is not None and {"JobLevel","MarketMedianCTC"}.issubset(set(bench_df.columns)):
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce").fillna(0)
        merged = emp_df.merge(bench_df[["JobLevel","MarketMedianCTC"]].drop_duplicates(), on="JobLevel", how="left")
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace(0,pd.NA))*100
        comp_summary = merged.groupby("JobLevel", observed=True)[["CTC","MarketMedianCTC","DiffPct"]].mean().reset_index().rename(columns={"CTC":"AvgCTC"})
        comp_summary = comp_summary.round(2)
        comp_melt = comp_summary.melt(id_vars="JobLevel", value_vars=["AvgCTC","MarketMedianCTC"], var_name="Type", value_name="Value")
        fig_market = px.bar(comp_melt, x="JobLevel", y="Value", color="Type", barmode="group", title="Internal vs Market Median by Level")
        fig_market.update_layout(template="plotly_white")

    blocks = [
        {"title":"CTC by Job Level","desc":"Average internal CTC by level","df":ctc_level.round(2),"fig":fig_ctc,"insights":[f"Average CTC: ₹{avg_ctc:,.0f}"]},
        {"title":"Bonus by Job Level","desc":"Average bonus % by level","df":bonus_level.round(2),"fig":fig_bonus,"insights":[f"Average bonus %: {avg_bonus_pct:.1f}%"]},
        {"title":"Gender Pay Gap","desc":"Avg CTC by gender","df":gender_gap.round(2),"fig":fig_gender,"insights":[]}
    ]
    if comp_summary is not None and fig_market is not None:
        blocks.append({"title":"Internal vs Market","desc":"Company vs market median comparison","df":comp_summary,"fig":fig_market,"insights":[]})

    return blocks

# ---------------------------
# 4) Workforce helper
# ---------------------------
def get_workforce_blocks(df: pd.DataFrame):
    if df is None or df.empty:
        return []

    required = ["EmployeeID","JobLevel","Gender"]
    for c in required:
        if c not in df.columns:
            return [{"title":"Workforce Data Missing","desc":f"Required column {c} not found","df":None,"fig":None,"insights":[]}]

    total = len(df)
    female_pct = (df["Gender"].astype(str).str.lower()=="female").mean()*100
    job_levels = df["JobLevel"].nunique()

    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount").sort_values("Headcount", ascending=True)
    fig_hc = px.bar(hc, x="Headcount", y="JobLevel", orientation="h", text="Headcount", title="Headcount by Job Level", color="JobLevel")
    fig_hc.update_layout(template="plotly_white")

    span_df = None
    fig_span = None
    if "ManagerID" in df.columns:
        manager_counts = df["ManagerID"].value_counts().reset_index()
        manager_counts.columns = ["ManagerID","DirectReports"]
        span_df = manager_counts
        fig_span = px.histogram(manager_counts, x="DirectReports", nbins=15, title="Distribution of Direct Reports per Manager")
        fig_span.update_layout(template="plotly_white")

    skills_df = None
    fig_skills = None
    if "Skills" in df.columns:
        from collections import Counter
        tokens = Counter()
        for v in df["Skills"].dropna().astype(str):
            parts = [x.strip().lower() for x in v.replace("|",",").split(",") if x.strip()]
            tokens.update(parts)
        if tokens:
            skills_df = pd.DataFrame(tokens.most_common(20), columns=["Skill","Count"])
            fig_skills = px.bar(skills_df.sort_values("Count", ascending=True), x="Count", y="Skill", orientation="h", title="Top 20 Skills")
            fig_skills.update_layout(template="plotly_white")

    blocks = [
        {"title":"Headcount Structure","desc":"Headcount distribution across job levels","df":hc,"fig":fig_hc,"insights":[f"Total employees: {total}", f"Female %: {female_pct:.1f}%"]},
    ]
    if span_df is not None:
        blocks.append({"title":"Manager Spans","desc":"Direct reports per manager","df":span_df,"fig":fig_span,"insights":[]})
    if skills_df is not None:
        blocks.append({"title":"Skill Inventory","desc":"Top skills across the workforce","df":skills_df,"fig":fig_skills,"insights":[]})

    return blocks

# ---------------------------
# 5) Engagement helper
# ---------------------------
def get_engagement_blocks(df: pd.DataFrame):
    if df is None or df.empty:
        return []

    # find question columns starting with Q
    qcols = [c for c in df.columns if str(c).strip().upper().startswith("Q")]
    if not qcols:
        return [{"title":"Engagement Data Missing","desc":"No survey question columns found (Q*)","df":None,"fig":None,"insights":[]}]

    df[qcols] = df[qcols].apply(pd.to_numeric, errors="coerce")
    df["EngagementIndex"] = df[qcols].mean(axis=1)

    avg_index = float(df["EngagementIndex"].mean())
    pct_high = float((df["EngagementIndex"] > 3.6).mean() * 100)
    pct_low = float((df["EngagementIndex"] <= 2.9).mean() * 100)

    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanIndex","MedianIndex","Count","StdDev"]
    dept_summary = dept_summary.round(2)

    gender_summary = df.groupby("Gender", observed=True)["EngagementIndex"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanIndex","Count"]
    gender_summary = gender_summary.round(2)

    bins = [-1, 2.9, 3.6, 5]
    labels = ["Low","Medium","High"]
    df["EngagementCat"] = pd.cut(df["EngagementIndex"], bins=bins, labels=labels).astype(str)
    cat_counts = df["EngagementCat"].value_counts().reindex(labels, fill_value=0).reset_index()
    cat_counts.columns = ["Category","Count"]

    fig_dept = px.bar(dept_summary.sort_values("MeanIndex", ascending=False), x="Department", y="MeanIndex", text="MeanIndex", title="Avg Engagement by Department")
    fig_dept.update_layout(template="plotly_white")
    fig_cat = px.pie(cat_counts, names="Category", values="Count", title="Engagement Categories")
    fig_cat.update_layout(template="plotly_white")
    fig_gender = px.bar(gender_summary, x="Gender", y="MeanIndex", text="MeanIndex", title="Avg Engagement by Gender")
    fig_gender.update_layout(template="plotly_white")

    blocks = [
        {"title":"Engagement Index Overview","desc":"Overall engagement index (mean of Q* responses).","df":pd.DataFrame([{"Metric":"AvgIndex","Value":round(avg_index,2)}, {"Metric":"Responses","Value":len(df)}]),"fig":None,"insights":[f"Average engagement index: {avg_index:.2f}", f"Highly engaged: {pct_high:.1f}%"]},
        {"title":"Departmental Engagement","desc":"Average engagement score by department.","df":dept_summary,"fig":fig_dept,"insights":[]},
        {"title":"Engagement Categories","desc":"High/Medium/Low segmentation of engagement.","df":cat_counts,"fig":fig_cat,"insights":[f"High engagement share: {pct_high:.1f}%"]},
        {"title":"Demographic Engagement","desc":"Engagement by gender.","df":gender_summary,"fig":fig_gender,"insights":[]}
    ]
    return blocks

# ---------------------------
# UI: Uploads & Generate
# ---------------------------
def run_consolidated_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#111827,#374151);color:white;">
      <h2 style="margin:0">📚 Consolidated HR Reporting Engine</h2>
      <p style="margin:4px 0 0 0;">Upload all datasets and generate a combined Executive PDF for leadership.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Upload the datasets (CSV/XLSX). You can skip optional benchmark. Filenames are shown to confirm upload.")

    col1, col2 = st.columns(2)
    with col1:
        perf_df = upload_data("Upload Performance Data (CSV/XLSX) — required")
        emp_comp_df = upload_data("Upload Compensation (internal) — required")
        workforce_df = upload_data("Upload Workforce Data (CSV/XLSX) — required")
    with col2:
        attr_df = upload_data("Upload Attrition Data (CSV/XLSX) — required")
        engagement_df = upload_data("Upload Engagement Survey (CSV/XLSX) — required")
        bench_df = upload_data("Upload Benchmark Data (optional)")

    # Validate at least one required dataset exists
    required_ok = all([not (x is None) for x in [perf_df, emp_comp_df, workforce_df, attr_df, engagement_df]])
    if not required_ok:
        st.warning("Please upload all required datasets before generating the consolidated report.")
        return

    st.markdown("---")
    if st.button("🧾 Generate Consolidated HR Executive PDF", use_container_width=True):
        # Gather blocks from each helper
        try:
            blocks = []
            blocks.extend(get_performance_blocks(perf_df))
            blocks.extend(get_attrition_blocks(attr_df))
            blocks.extend(get_compensation_blocks(emp_comp_df, bench_df))
            blocks.extend(get_workforce_blocks(workforce_df))
            blocks.extend(get_engagement_blocks(engagement_df))

            # Basic sanity: filter out empty blocks
            final_blocks = [b for b in blocks if b and (b.get("df") is not None or b.get("fig") is not None or b.get("insights"))]

            if not final_blocks:
                st.error("No content to generate consolidated report.")
                return

            # Single call to pdf helper with combined blocks
            render_pdf_download_button("Consolidated HR Executive Report", "Consolidated Modules", final_blocks, "Consolidated_HR_Report")
        except Exception as e:
            st.exception(e)


# For page auto-discovery (if you use pages, import and call run_consolidated_module)
if __name__ == "__main__":
    run_consolidated_module()