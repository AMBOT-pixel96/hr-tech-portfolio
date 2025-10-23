import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image

MODULE_COLOR = "#22C55E"  # green

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def run_compensation_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;
         background:linear-gradient(90deg,#14532D,#22C55E);color:white;">
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

    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce").fillna(0)
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce").fillna(0)
    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"].replace({0: pd.NA})) * 100
    emp_df["BonusPct"] = emp_df["BonusPct"].fillna(0)

    avg_ctc = float(emp_df["CTC"].mean())
    avg_bonus_pct = float(emp_df["BonusPct"].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg CTC", f"₹{avg_ctc:,.0f}")
    c2.metric("Avg Bonus %", f"{avg_bonus_pct:.1f}%")
    c3.metric("Employee Count", len(emp_df))

    ctc_level = _round_df(emp_df.groupby("JobLevel", observed=True)["CTC"].mean()
                          .reset_index().rename(columns={"CTC":"AvgCTC"}))
    bonus_level = _round_df(emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean()
                            .reset_index().rename(columns={"BonusPct":"AvgBonusPct"}))

    fig_ctc = px.bar(ctc_level, x="JobLevel", y="AvgCTC", text="AvgCTC",
                     title="Avg CTC by Job Level", color="JobLevel", template="plotly_white")
    fig_ctc.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    fig_bonus = px.bar(bonus_level, x="JobLevel", y="AvgBonusPct", text="AvgBonusPct",
                       title="Avg Bonus % by Job Level", color="JobLevel", template="plotly_white")
    fig_bonus.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color='black', marker_line_width=1)

    gender_gap = _round_df(emp_df.groupby("Gender", observed=True)["CTC"].mean()
                           .reset_index().rename(columns={"CTC":"AvgCTC"}))
    fig_gender = px.bar(gender_gap, x="Gender", y="AvgCTC", text="AvgCTC",
                        title="Avg CTC by Gender", color="Gender", template="plotly_white")
    fig_gender.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    comp_summary = None
    fig_market = None
    if bench_df is not None and {"JobLevel","MarketMedianCTC"}.issubset(set(bench_df.columns)):
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce").fillna(0)
        merged = emp_df.merge(bench_df[["JobLevel","MarketMedianCTC"]].drop_duplicates(),
                              on="JobLevel", how="left")
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) /
                             merged["MarketMedianCTC"].replace(0, pd.NA)) * 100
        comp_summary = _round_df(merged.groupby("JobLevel", observed=True)
                                 [["CTC","MarketMedianCTC","DiffPct"]]
                                 .mean().reset_index()
                                 .rename(columns={"CTC":"AvgCTC"}))
        comp_melt = comp_summary.melt(id_vars="JobLevel",
                                      value_vars=["AvgCTC","MarketMedianCTC"],
                                      var_name="Type", value_name="Value")
        fig_market = px.bar(comp_melt, x="JobLevel", y="Value", color="Type",
                            barmode="group", title="Internal vs Market Median by Level",
                            template="plotly_white")

    # Save chart images
    ctc_path = save_chart_image("CTC by Job Level", fig_ctc)
    bonus_path = save_chart_image("Bonus by Job Level", fig_bonus)
    gender_path = save_chart_image("Gender Pay Gap", fig_gender)
    market_path = save_chart_image("Internal vs Market", fig_market) if fig_market else None

    # App visuals
    st.subheader("Compensation by Job Level")
    st.dataframe(ctc_level, use_container_width=True)
    st.plotly_chart(fig_ctc, use_container_width=True)
    st.subheader("Bonus by Job Level")
    st.dataframe(bonus_level, use_container_width=True)
    st.plotly_chart(fig_bonus, use_container_width=True)
    st.subheader("Gender Pay Gap")
    st.dataframe(gender_gap, use_container_width=True)
    st.plotly_chart(fig_gender, use_container_width=True)
    if fig_market:
        st.subheader("Internal vs Market")
        st.dataframe(comp_summary, use_container_width=True)
        st.plotly_chart(fig_market, use_container_width=True)

    data_blocks = [
        {"title":"CTC by Job Level","desc":"Average internal CTC by level",
         "df":ctc_level,"fig_path":ctc_path,
         "insights":[f"Average CTC: ₹{avg_ctc:,.0f}"]},
        {"title":"Bonus by Job Level","desc":"Average bonus % by level",
         "df":bonus_level,"fig_path":bonus_path,
         "insights":[f"Average bonus %: {avg_bonus_pct:.1f}%"]},
        {"title":"Gender Pay Gap","desc":"Average CTC by gender",
         "df":gender_gap,"fig_path":gender_path,"insights":[]}
    ]
    if comp_summary is not None and market_path:
        data_blocks.append({"title":"Internal vs Market","desc":"Company vs market median comparison",
                            "df":comp_summary,"fig_path":market_path,"insights":[]})

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Compensation Analytics Executive Report",
                               "Compensation", data_blocks, "Compensation")