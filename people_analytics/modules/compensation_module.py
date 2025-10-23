# ============================================
# modules/compensation_module.py — v3.0.1 | Executive (Dual-theme safe + sync save)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image
from utils.fix_helper import ensure_chart_saved

MODULE_COLOR = "#22C55E"

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float","int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def run_compensation_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#14532D,#22C55E);color:white;">
      <h2 style="margin:0">💰 Compensation Analytics</h2>
      <p style="margin:4px 0 0 0;">Pay structure, bonus trends, gender parity & market benchmarking.</p>
    </div>
    """, unsafe_allow_html=True)

    emp_df = upload_data("Upload Internal Compensation Data (CSV/XLSX)")
    bench_df = upload_data("Upload Benchmark Data (optional)")
    if emp_df is None:
        return

    required = ["EmployeeID","Gender","Department","JobRole","JobLevel","CTC","Bonus"]
    missing = [c for c in required if c not in emp_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    # Numeric conversions & derived
    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce").fillna(0)
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce").fillna(0)
    emp_df["BonusPct"] = np.where(emp_df["CTC"] > 0, (emp_df["Bonus"] / emp_df["CTC"]) * 100, 0)

    avg_ctc = float(emp_df["CTC"].mean())
    avg_bonus_pct = float(emp_df["BonusPct"].mean())

    c1,c2,c3 = st.columns(3)
    c1.metric("Avg CTC", f"₹{avg_ctc:,.0f}")
    c2.metric("Avg Bonus %", f"{avg_bonus_pct:.1f}%")
    c3.metric("Employee Count", len(emp_df))

    # Aggregations
    ctc_level = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index().rename(columns={"CTC":"AvgCTC"})
    bonus_level = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index().rename(columns={"BonusPct":"AvgBonusPct"})
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index().rename(columns={"CTC":"AvgCTC"})

    ctc_level = _round_df(ctc_level)
    bonus_level = _round_df(bonus_level)
    gender_gap = _round_df(gender_gap)

    # Figures — use white template for consistent PDF colors
    fig_ctc = px.bar(ctc_level, x="JobLevel", y="AvgCTC", text="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
    fig_ctc.update_layout(template="plotly_white")
    fig_ctc.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    fig_bonus = px.bar(bonus_level, x="JobLevel", y="AvgBonusPct", text="AvgBonusPct", color="JobLevel", title="Avg Bonus % by Job Level")
    fig_bonus.update_layout(template="plotly_white")
    fig_bonus.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color='black', marker_line_width=1)

    fig_gender = px.bar(gender_gap, x="Gender", y="AvgCTC", text="AvgCTC", color="Gender", title="Avg CTC by Gender")
    fig_gender.update_layout(template="plotly_white")
    fig_gender.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    # Market benchmark (optional)
    comp_summary = None
    fig_market = None
    if bench_df is not None and {"JobLevel","MarketMedianCTC"}.issubset(set(bench_df.columns)):
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce").fillna(0)
        merged = emp_df.merge(bench_df[["JobLevel","MarketMedianCTC"]].drop_duplicates(), on="JobLevel", how="left")
        merged["DiffPct"] = np.where(merged["MarketMedianCTC"]>0, (merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"] * 100, 0)
        comp_summary = merged.groupby("JobLevel", observed=True)[["CTC","MarketMedianCTC","DiffPct"]].mean().reset_index().rename(columns={"CTC":"AvgCTC"})
        comp_summary = _round_df(comp_summary)
        comp_melt = comp_summary.melt(id_vars="JobLevel", value_vars=["AvgCTC","MarketMedianCTC"], var_name="Type", value_name="Value")
        fig_market = px.bar(comp_melt, x="JobLevel", y="Value", color="Type", barmode="group", title="Internal vs Market Median by Level")
        fig_market.update_layout(template="plotly_white")
        fig_market.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    # App display
    st.subheader("Compensation by Job Level")
    st.dataframe(ctc_level, use_container_width=True)
    st.plotly_chart(fig_ctc, use_container_width=True)

    st.subheader("Bonus by Job Level")
    st.dataframe(bonus_level, use_container_width=True)
    st.plotly_chart(fig_bonus, use_container_width=True)

    st.subheader("Gender Pay Gap")
    st.dataframe(gender_gap, use_container_width=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    if fig_market is not None:
        st.subheader("Internal vs Market")
        st.dataframe(comp_summary, use_container_width=True)
        st.plotly_chart(fig_market, use_container_width=True)

    # Save images synchronously (ensures images are present before PDF compile)
    # using ensure_chart_saved helper that will call save_chart_image and wait/validate
    saved_assets = {}
    for title, fig in [
        ("Avg CTC by Job Level", fig_ctc),
        ("Avg Bonus by Job Level", fig_bonus),
        ("Avg CTC by Gender", fig_gender),
        ("Internal vs Market", fig_market)
    ]:
        if fig is not None:
            path = ensure_chart_saved(fig, title, save_chart_image)
            if path:
                saved_assets[title] = {"png": {"path": path}}

    # data blocks for PDF
    data_blocks = [
        {"title":"CTC by Job Level","desc":"Average internal CTC by level","df":ctc_level,"fig":fig_ctc,
         "insights":[f"Average CTC: ₹{avg_ctc:,.0f}"], "asset": saved_assets.get("Avg CTC by Job Level")},
        {"title":"Bonus by Job Level","desc":"Average bonus % by level","df":bonus_level,"fig":fig_bonus,
         "insights":[f"Average bonus %: {avg_bonus_pct:.1f}%"], "asset": saved_assets.get("Avg Bonus by Job Level")},
        {"title":"Gender Pay Gap","desc":"Avg CTC by gender","df":gender_gap,"fig":fig_gender,
         "insights":[], "asset": saved_assets.get("Avg CTC by Gender")},
    ]
    if comp_summary is not None and fig_market is not None:
        data_blocks.append({"title":"Internal vs Market","desc":"Company vs market median comparison","df":comp_summary,"fig":fig_market,"insights":[],"asset": saved_assets.get("Internal vs Market")})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Compensation Analytics Executive Report", "Compensation", data_blocks, "Compensation")