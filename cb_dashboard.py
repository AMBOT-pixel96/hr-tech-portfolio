# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 4.3 (UAT-3 Final Polish)
# Last Updated: 2025-09-30
# Notes:
# - UAT-3 Observations polished:
#   * Fixed quartile + gender table layouts
#   * Consistent palette across all charts
#   * Chart titles standardized
#   * PDF tables spread evenly with colWidths
#   * Gender gap % visible in dashboard + PDF
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import os

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Compensation & Benefits Dashboard", layout="wide")
TMP_DIR = "temp_charts_cb"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Required headers
# -----------------------
EMP_REQUIRED = [
    "EmployeeID", "Gender", "Department", "JobRole",
    "JobLevel", "CTC", "Bonus", "PerformanceRating"
]
BENCH_REQUIRED = ["JobRole", "JobLevel", "MarketMedianCTC"]

# -----------------------
# Visual / PDF constants
# -----------------------
PALETTE = px.colors.qualitative.Prism
TABLE_ZEBRA = colors.HexColor("#F7F7F7")
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TEXT_COLOR = colors.black
# -----------------------
# Helpers
# -----------------------
def validate_exact_headers(df_or_cols, required_cols):
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return (cols == required_cols, "OK" if cols == required_cols else f"Header mismatch. Expected {required_cols}, found {cols}")

def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def safe_filename(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def readable_lakhs_number(x):
    if pd.isna(x): return None
    try: return round(float(x) / 100000.0, 2)
    except: return None

def draw_background(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.black)
    canvas.rect(5, 5, A4[0]-10, A4[1]-10, stroke=1, fill=0)
    canvas.restoreState()

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

def save_plotly_asset(fig, filename_base, width=1200, height=700, scale=2):
    base = os.path.join(TMP_DIR, filename_base)
    png_path, html_path = base + ".png", base + ".html"
    try:
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.update_layout(template="plotly_white", title_font=dict(size=18, color="black", family="Helvetica"))
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f: f.write(img_bytes)
        return {"png": png_path, "html": None}
    except Exception:
        try:
            fig.write_html(html_path)
            return {"png": None, "html": html_path}
        except Exception:
            return {"png": None, "html": None}

def apply_chart_style(fig, title: str):
    """Apply consistent hybrid 'Nightfall Neutral' styling (optimized for both dark/light modes)."""
    bg_color = "#0f172a"     # dark slate
    text_color = "#e2e8f0"   # soft gray text
    grid_color = "#475569"   # muted gridlines
    accent_palette = px.colors.qualitative.Set2

    fig.update_layout(
        title=dict(text=title, x=0.45, xanchor="center", yanchor="top"),
        template="plotly_white",
        title_font=dict(size=20, color=text_color, family="Helvetica-Bold"),
        font=dict(color=text_color, family="Helvetica", size=12),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(
            title_font=dict(size=13, color=text_color),
            tickfont=dict(size=11, color=text_color),
            gridcolor=grid_color,
            showline=True,
            linecolor=grid_color,
            mirror=True
        ),
        yaxis=dict(
            title_font=dict(size=13, color=text_color),
            tickfont=dict(size=11, color=text_color),
            gridcolor=grid_color,
            showline=True,
            linecolor=grid_color,
            mirror=True
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=text_color),
            bgcolor=bg_color,
            bordercolor=grid_color,
            borderwidth=0.5
        ),
        margin=dict(t=60, l=50, r=50, b=110)
    )

    # ensure decent bar visibility and spacing
    fig.update_traces(marker=dict(line=dict(width=0.3, color="#1e293b")), width=0.5)
    fig.update_layout(colorway=accent_palette)
    return fig
# -----------------------
# Templates + How-to Guide
# -----------------------
def get_employee_template_csv():
    df = pd.DataFrame(columns=EMP_REQUIRED)
    df.loc[0] = ["E1001", "Male", "Finance", "Analyst", "Analyst", 600000, 50000, 3]
    return df.to_csv(index=False)

def get_benchmark_template_csv():
    df = pd.DataFrame(columns=BENCH_REQUIRED)
    df.loc[0] = ["Analyst", "Analyst", 650000]
    return df.to_csv(index=False)

# -----------------------
# App header
# -----------------------
st.markdown(f"""
<div style="padding:18px;border-radius:10px;border:1px solid #ddd;text-align:center">
  <h1 style="margin:0;padding:0;font-size:30px;color:#4B0082">📊 Compensation & Benefits Dashboard</h1>
  <p style="font-size:14px;">Board-ready pay analytics — per-metric filters, exports, and benchmarks.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Step 1: Templates & Guide
# -----------------------
st.header("Step 1 — Templates & Guide")
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Internal Compensation Data Template",
                       data=get_employee_template_csv(),
                       file_name="Internal_Compensation_Data_Template.csv")
with c2:
    st.download_button("📥 External Benchmarking Data Template",
                       data=get_benchmark_template_csv(),
                       file_name="External_Benchmarking_Data_Template.csv")

if not st.checkbox("✅ I downloaded templates + guide"):
    st.stop()
# -----------------------
# Step 2: Upload Data
# -----------------------
st.header("Step 2 — Upload Data")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Internal Compensation CSV/XLSX", type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("Upload External Benchmarking CSV/XLSX", type=["csv","xlsx"])

if not uploaded_file: st.stop()

def read_input(file):
    return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file, engine="openpyxl")

emp_df = read_input(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok: st.error(msg); st.stop()

bench_df = None
if benchmark_file:
    bench_df = read_input(benchmark_file)
    ok_b, msg_b = validate_exact_headers(bench_df, BENCH_REQUIRED)
    if not ok_b: st.error(msg_b); st.stop()

# -----------------------
# Filters per-metric
# -----------------------
def metric_filters_ui(df, prefix=""):
    st.markdown("**Filters (for this metric only):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        dept = st.selectbox("Department", ["All"]+sorted(df["Department"].dropna().unique()), key=f"{prefix}_dept")
    with c2:
        roles = sorted(df[df["Department"]==dept]["JobRole"].unique()) if dept!="All" else sorted(df["JobRole"].unique())
        sel_roles = st.multiselect("Job Role", roles, key=f"{prefix}_roles")
    with c3:
        levels = sorted(df["JobLevel"].dropna().unique())
        sel_levels = st.multiselect("Job Level", levels, key=f"{prefix}_levels")
    out = df.copy()
    if dept!="All": out=out[out["Department"]==dept]
    if sel_roles: out=out[out["JobRole"].isin(sel_roles)]
    if sel_levels: out=out[out["JobLevel"].isin(sel_levels)]
    return out

# -----------------------
# Metrics storage
# -----------------------
sections=[]; images_for_download=[]

# -----------------------
# Metric A: Average CTC by Job Level
# -----------------------
st.subheader("🏷️ Average CTC by Job Level")
dfA = metric_filters_ui(emp_df, prefix="A")

avg = dfA.groupby("JobLevel")["CTC"].agg(TotalCTC="sum", AverageCTC="mean").reset_index()
avg["Total CTC (Cr.)"] = (avg["TotalCTC"] / 1e7).round(2)
avg["Average CTC (₹ Lakhs)"] = avg["AverageCTC"].apply(readable_lakhs_number)

st.dataframe(avg[["JobLevel", "Total CTC (Cr.)", "Average CTC (₹ Lakhs)"]])

figA = px.bar(avg, x="JobLevel", y="AverageCTC", color="JobLevel",
              color_discrete_sequence=px.colors.qualitative.Set2,
              labels={"AverageCTC": "Average CTC (₹)"})
figA = apply_chart_style(figA, "Average CTC by Job Level")
assetA = save_plotly_asset(figA, safe_filename("avg_ctc"))
st.plotly_chart(figA, use_container_width=True)

sections.append(("Average CTC by Job Level",
                 "Average and total pay across job levels.",
                 avg[["JobLevel", "Total CTC (Cr.)", "Average CTC (₹ Lakhs)"]],
                 assetA))


# -----------------------
# Metric B: Median CTC by Job Level
# -----------------------
st.subheader("📏 Median CTC by Job Level")
dfB = metric_filters_ui(emp_df, prefix="B")

med = dfB.groupby("JobLevel")["CTC"].agg(TotalCTC="sum", MedianCTC="median").reset_index()
med["Total CTC (Cr.)"] = (med["TotalCTC"] / 1e7).round(2)
med["Median CTC (₹ Lakhs)"] = med["MedianCTC"].apply(readable_lakhs_number)
st.dataframe(med[["JobLevel", "Total CTC (Cr.)", "Median CTC (₹ Lakhs)"]])

figB = px.bar(med, x="JobLevel", y="MedianCTC", color="JobLevel",
              color_discrete_sequence=px.colors.qualitative.Set2,
              labels={"MedianCTC": "Median CTC (₹)"})
figB = apply_chart_style(figB, "Median CTC by Job Level")
assetB = save_plotly_asset(figB, safe_filename("median_ctc"))
st.plotly_chart(figB, use_container_width=True)


# -----------------------
# Metric C: Quartile Distribution
# -----------------------
st.subheader("📉 Quartile Distribution")
dfC = metric_filters_ui(emp_df, prefix="C")
rows = []
for lvl,g in dfC.groupby("JobLevel"):
    vc = pd.qcut(g["CTC"], 4, labels=["Q1","Q2","Q3","Q4"]).value_counts(normalize=True)*100
    for q,v in vc.items():
        rows.append({"JobLevel":lvl,"Quartile":q,"Share%":round(v,2)})
quart_tbl = pd.DataFrame(rows).pivot(index="JobLevel",columns="Quartile",values="Share%").reset_index().fillna("")
st.dataframe(quart_tbl)
figC = px.pie(pd.DataFrame(rows), names="Quartile", values="Share%",
              color="Quartile", color_discrete_sequence=px.colors.qualitative.Set2,
              hole=0.5)
figC = apply_chart_style(figC, "Quartile Distribution (Share of Employees)")
assetC = save_plotly_asset(figC, safe_filename("quartile_donut"))
st.plotly_chart(figC, use_container_width=True)


# -----------------------
# Metric D: Bonus % of CTC
# -----------------------
st.subheader("🎁 Bonus % of CTC by Job Level")
dfD = metric_filters_ui(emp_df, prefix="D")
dfD["Bonus %"] = np.where(dfD["CTC"] > 0, (dfD["Bonus"] / dfD["CTC"]) * 100, np.nan)
bonus = dfD.groupby("JobLevel")["Bonus %"].mean().reset_index().round(2)
st.dataframe(bonus)
figD = px.bar(bonus, x="JobLevel", y="Bonus %", color="JobLevel",
              color_discrete_sequence=px.colors.qualitative.Set2,
              labels={"Bonus %": "Avg Bonus (%)"})
figD = apply_chart_style(figD, "Average Bonus % of CTC by Job Level")
assetD = save_plotly_asset(figD, safe_filename("bonus_pct"))
st.plotly_chart(figD, use_container_width=True)


# -----------------------
# Metric E: Company vs Market
# -----------------------
if bench_df is not None:
    st.subheader("🏛️ Company vs Market (Median CTC)")
    dfE = metric_filters_ui(emp_df, prefix="E")
    comp = dfE.groupby("JobLevel")["CTC"].median().reset_index().rename(columns={"CTC": "CompanyMedian"})
    bench = bench_df.groupby("JobLevel")["MarketMedianCTC"].median().reset_index()
    compare = pd.merge(comp, bench, on="JobLevel", how="outer")
    compare["Gap %"] = np.where(compare["MarketMedianCTC"]>0, 
                                (compare["CompanyMedian"]-compare["MarketMedianCTC"])/compare["MarketMedianCTC"]*100, np.nan).round(2)
    compare["Company (₹ Lakhs)"] = compare["CompanyMedian"].apply(readable_lakhs_number)
    compare["Market (₹ Lakhs)"] = compare["MarketMedianCTC"].apply(readable_lakhs_number)
    st.dataframe(compare)

    figE = go.Figure()
    figE.add_trace(go.Bar(x=compare["JobLevel"], y=compare["CompanyMedian"], name="Company", marker_color="#3b82f6"))
    figE.add_trace(go.Scatter(x=compare["JobLevel"], y=compare["MarketMedianCTC"], name="Market",
                              mode="lines+markers", marker=dict(color="#f87171", size=8), line=dict(width=2)))
    figE = apply_chart_style(figE, "Company vs Market (Median CTC ₹ Lakhs)")
    assetE = save_plotly_asset(figE, safe_filename("cmp_vs_market"))
    st.plotly_chart(figE, use_container_width=True)


# -----------------------
# Metric F: Avg CTC by Gender & Job Level
# -----------------------
st.subheader("👫 Average CTC by Gender & Job Level")
dfF = metric_filters_ui(emp_df, prefix="F")
g = dfF.groupby(["JobLevel", "Gender"])["CTC"].mean().reset_index()
g["Lakhs"] = g["CTC"].apply(readable_lakhs_number)
st.dataframe(g.pivot(index="JobLevel", columns="Gender", values="Lakhs").fillna(""))
figF = px.bar(g, x="JobLevel", y="Lakhs", color="Gender", barmode="group",
              color_discrete_sequence=px.colors.qualitative.Set2)
figF = apply_chart_style(figF, "Average CTC by Gender & Job Level")
assetF = save_plotly_asset(figF, safe_filename("gender_ctc"))
st.plotly_chart(figF, use_container_width=True)


# -----------------------
# Metric G: Avg CTC by Performance Rating & Job Level
# -----------------------
st.subheader("⭐ Average CTC by Performance Rating & Job Level")
dfG = metric_filters_ui(emp_df, prefix="G")
r = dfG.groupby(["JobLevel", "PerformanceRating"])["CTC"].mean().reset_index()
r["Lakhs"] = r["CTC"].apply(readable_lakhs_number)
r["PerformanceRating"] = r["PerformanceRating"].astype(str)
st.dataframe(r.pivot(index="JobLevel", columns="PerformanceRating", values="Lakhs").fillna(""))
figG = px.bar(r, x="JobLevel", y="Lakhs", color="PerformanceRating", barmode="group",
              color_discrete_sequence=px.colors.qualitative.Set2,
              labels={"Lakhs": "Avg CTC (₹ Lakhs)", "PerformanceRating": "Rating"})
figG = apply_chart_style(figG, "Average CTC by Performance Rating & Job Level")
assetG = save_plotly_asset(figG, safe_filename("rating_ctc"))
st.plotly_chart(figG, use_container_width=True)
# -----------------------
# Compiled PDF Report
# -----------------------
class PDFBookmark(Flowable):
    def __init__(self, name, title):
        super().__init__()
        self.name, self.title = name, title
    def wrap(self, availWidth, availHeight): return (0,0)
    def draw(self):
        try:
            self.canv.bookmarkPage(self.name)
            self.canv.addOutlineEntry(self.title, self.name, level=0, closed=False)
        except Exception: pass

st.header("📥 Download Reports")
st.write("Choose metrics to include in the compiled PDF:")
kpi_check = {title: st.checkbox(title, key=f"chk_{i}") for i,(title,_,_,_) in enumerate(sections)}

if st.button("🧾 Compile Selected Report"):
    selected_titles = [t for t,v in kpi_check.items() if v]
    if not selected_titles:
        st.warning("Select at least one metric to include.")
    else:
        selected_sections = [s for s in sections if s[0] in selected_titles]
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        h1, h2, normal = styles["Title"], styles["Heading2"], styles["Normal"]
        body = ParagraphStyle("body", parent=normal, fontName=BODY_FONT, fontSize=10, leading=13)

        story = []
        cover_title = "<para align=center><font size=28 color='#4B0082'><b>Compensation & Benefits Report</b></font></para>"
        story.append(Paragraph(cover_title, body))
        story.append(Spacer(1, 18))
        story.append(Paragraph(f"<para align=center>Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}</para>", body))
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        story.append(Paragraph("Table of Contents", h2))
        for idx,(title,_,_,_) in enumerate(selected_sections,1):
            story.append(Paragraph(f"{idx}. {title}", body))
        story.append(PageBreak())

        for title,desc,tbl,asset in selected_sections:
            bname=sanitize_anchor(title)
            story.append(PDFBookmark(bname,title))
            story.append(Paragraph(f"<b>{title}</b>", h2)); story.append(Spacer(1,6))
            if desc: story.append(Paragraph(desc, body)); story.append(Spacer(1,6))
            if tbl is not None and hasattr(tbl,"shape") and tbl.shape[0]>0:
                data=[list(tbl.columns)]+tbl.fillna("").values.tolist()
                colWidths=[(A4[0]-40)/len(tbl.columns)]*len(tbl.columns)
                t=Table(data,repeatRows=1,hAlign="LEFT",colWidths=colWidths)
                tstyle=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                                   ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
                                   ("VALIGN",(0,0),(-1,-1),"MIDDLE")])
                for r in range(1,len(data)):
                    if r%2==0: tstyle.add("BACKGROUND",(0,r),(-1,r),TABLE_ZEBRA)
                t.setStyle(tstyle); story.append(t); story.append(Spacer(1,8))
            if asset:
                if asset.get("png") and os.path.exists(asset["png"]):
                    story.append(RLImage(asset["png"],width=170*mm,height=90*mm)); story.append(Spacer(1,6))
            story.append(Paragraph(f"<i>Insight:</i> Review {title} for trends.", body))
            story.append(PageBreak())

        doc.build(story,onFirstPage=lambda c,d:(draw_background(c,d),add_page_number(c,d)),
                         onLaterPages=lambda c,d:(draw_background(c,d),add_page_number(c,d)))
        st.download_button("⬇️ Download Compiled PDF", buf.getvalue(),
                           file_name="cb_dashboard_compiled.pdf", mime="application/pdf")

# -----------------------
# Quick Downloads + Wrap
# -----------------------
st.subheader("📸 Quick Chart Downloads")
for item in images_for_download:
    title,asset=item.get("title","chart"),item.get("asset",{})
    if asset.get("png") and os.path.exists(asset["png"]):
        with open(asset["png"],"rb") as f:
            st.download_button(f"⬇️ {title} (PNG)",f.read(),
                               file_name=os.path.basename(asset["png"]),mime="image/png")
    elif asset.get("html") and os.path.exists(asset["html"]):
        with open(asset["html"],"rb") as f:
            st.download_button(f"⬇️ {title} (HTML)",f.read(),
                               file_name=os.path.basename(asset["html"]),mime="text/html")

st.success("Dashboard loaded ✅ V4.3 ready: clean tables, consistent charts, gender gap %, PDF polish.")

# -------------------------------
# Enhancement - Chatbot Assistant Add-on (v3 Clean)
# -------------------------------

def safe_markdown_table(df):
    """Try to render df as markdown table, fallback to st.dataframe if tabulate missing."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        st.warning("⚠️ Markdown table requires `tabulate` (not installed). Showing as dataframe instead.")
        st.dataframe(df)
        return None

def run_chatbot_ui():
    """Buffed rule-based chatbot for C&B Dashboard (free-tier)."""
    st.subheader("💬 C&B Data Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about CTC, Bonus %, Gender Gap, Market vs Company..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        query = prompt.lower()

        response = "🤔 I didn’t catch that. Try asking about **CTC levels**, **bonus %**, **gender gap**, or **market comparison**."

        # === Metric A: Avg CTC ===
        if any(word in query for word in ["average ctc", "mean ctc", "avg salary"]):
            avg = emp_df.groupby("JobLevel")["CTC"].mean().reset_index()
            avg["Lakhs"] = avg["CTC"].apply(readable_lakhs_number)
            insight = f"👉 Directors earn ~{avg['Lakhs'].max()}L vs Analysts ~{avg['Lakhs'].min()}L."
            table_md = safe_markdown_table(avg)
            if table_md:
                response = f"📊 **Average CTC by JobLevel:**\n\n{table_md}\n\n**Insight:** {insight}"
            st.plotly_chart(px.bar(avg, x="JobLevel", y="Lakhs", title="Average CTC by Level", color="JobLevel"))

        # === Metric F: Gender Gap ===
        elif "gender" in query or "pay gap" in query:
            g = emp_df.groupby(["JobLevel","Gender"])["CTC"].mean().reset_index()
            g["Lakhs"] = g["CTC"].apply(readable_lakhs_number)
            pivot = g.pivot(index="JobLevel", columns="Gender", values="Lakhs").reset_index().fillna("")
            table_md = safe_markdown_table(pivot)
            if table_md:
                response = f"👫 **Gender Pay Gap (Lakhs):**\n\n{table_md}\n\n**Insight:** Male vs Female CTC gap is {round((pivot['Male'].mean()-pivot['Female'].mean())/pivot['Female'].mean()*100,1)}% overall."
            st.plotly_chart(px.bar(g, x="JobLevel", y="Lakhs", color="Gender", barmode="group", title="Gender Gap by Level"))

        # === Metric D: Bonus % ===
        elif "bonus" in query:
            dfD = emp_df.assign(**{"Bonus %": np.where(emp_df["CTC"] > 0, (emp_df["Bonus"]/emp_df["CTC"])*100, np.nan)})
            bonus = dfD.groupby("JobLevel")["Bonus %"].mean().reset_index()
            bonus["Bonus %"] = bonus["Bonus %"].round(2)
            insight = f"🎁 Highest bonus % at {bonus.loc[bonus['Bonus %'].idxmax(), 'JobLevel']} level."
            table_md = safe_markdown_table(bonus)
            if table_md:
                response = f"🎁 **Bonus % of CTC by JobLevel:**\n\n{table_md}\n\n**Insight:** {insight}"
            st.plotly_chart(px.bar(bonus, x="JobLevel", y="Bonus %", title="Bonus % by Level", color="JobLevel"))

        # === Metric E: Market vs Company ===
        elif "market" in query or "comparison" in query:
            if bench_df is not None:
                comp = emp_df.groupby("JobLevel")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
                bench = bench_df.groupby("JobLevel")["MarketMedianCTC"].median().reset_index()
                compare = pd.merge(comp, bench, on="JobLevel", how="outer")
                compare["Gap %"] = np.where(compare["MarketMedianCTC"]>0, (compare["CompanyMedian"]-compare["MarketMedianCTC"])/compare["MarketMedianCTC"]*100, np.nan).round(2)
                insight = f"📉 Biggest negative gap is at {compare.loc[compare['Gap %'].idxmin(), 'JobLevel']} level."
                table_md = safe_markdown_table(compare)
                if table_md:
                    response = f"📉 **Company vs Market (Median CTC):**\n\n{table_md}\n\n**Insight:** {insight}"
                fig = go.Figure()
                fig.add_trace(go.Bar(x=compare["JobLevel"], y=compare["CompanyMedian"], name="Company"))
                fig.add_trace(go.Scatter(x=compare["JobLevel"], y=compare["MarketMedianCTC"], name="Market", mode="lines+markers"))
                st.plotly_chart(fig)
            else:
                response = "⚠️ Please upload a benchmark dataset to compare against the market."

        # Save + show
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
# Sidebar toggle
st.sidebar.subheader("🤖 Chatbot Assistant")
chat_mode = st.sidebar.checkbox("Enable Chatbot", value=False)
if chat_mode:
    run_chatbot_ui()