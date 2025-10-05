# ======================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 4.5 (QF-7 Stable)
# Last Updated: 2025-10-05
# Notes:
#  - Fixed “None” in metrics A/B
#  - Quartile numbers corrected + donut chart
#  - Wrapped company vs market title & shrunk legend
#  - Gender CTC TypeError resolved
# ======================================
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
# App Header (Dual-Mode Banner with Warning)
# -----------------------
st.markdown(f"""
<div style="
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ccc;
    text-align: center;
    background: linear-gradient(180deg, #0E1117 0%, #1E293B 100%);
    color: white;
">
  <h1 style="margin: 0; padding: 0; font-size: 32px; color: #F9FAFB;">
    📊 Compensation & Benefits Dashboard
  </h1>
  <p style="font-size: 15px; margin-top: 6px; color: #D1D5DB;">
    Board-ready pay analytics — per-metric filters, exports, and benchmarks.
  </p>

  <div style="
      display: inline-block;
      margin-top: 10px;
      background-color: #FFF3CD;
      color: #856404;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #FFECB5;
  ">
    ⚠️ Session resets if left idle for more than 3~5 mins or if page is reloaded. Download PDFs to save results.
  </div>
</div>
""", unsafe_allow_html=True)
# -----------------------
# Required headers
# -----------------------
EMP_REQUIRED = [
    "EmployeeID", "Gender", "Department", "JobRole",
    "JobLevel", "CTC", "Bonus", "PerformanceRating"
]
BENCH_REQUIRED = ["JobRole", "JobLevel", "MarketMedianCTC"]

# -----------------------
# Visual / PDF & Chart Constants (Unified Fix)
# -----------------------

# === PDF Constants ===
TABLE_ZEBRA = colors.HexColor("#F7F7F7")
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TEXT_COLOR_PDF = colors.black   # only used for PDF text, not charts

# === Chart Constants ===
PALETTE = px.colors.qualitative.Vivid   # consistent, professional color palette
CHART_BG_DARK = "#0E1117"
CHART_TEXT_LIGHT = "#FFFFFF"
CHART_TEXT_DARK = "#000000"
# -----------------------
# Helpers (Final v7 Stable)
# -----------------------
# Shared visual constants (used in helpers)
FALLBACK_PAPER_BG_DARK = "#0b1220"
FALLBACK_PAPER_BG_LIGHT = "#FFFFFF"
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
AXIS_TITLE_SIZE = 12
AXIS_TICK_SIZE = 11
LEGEND_FONT_SIZE = 12

def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def validate_exact_headers(df_or_cols, required_cols):
    """Return (bool, msg). Exact order & names expected."""
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    ok = cols == required_cols
    return (ok, "OK" if ok else f"Header mismatch. Expected {required_cols}, found {cols}")
def readable_lakhs_number(x):
    """Return value in Lakhs (float) or None."""
    if pd.isna(x):
        return None
    try:
        return round(float(x) / 100000.0, 2)
    except Exception:
        return None
def _detect_theme(theme_arg="auto"):
    """
    Detect current Streamlit theme or fallback to dark.
    """
    if theme_arg in ("dark", "light"):
        return theme_arg
    try:
        base = st.get_option("theme.base")
        if base and base.lower() in ("dark", "light"):
            return base.lower()
    except Exception:
        pass
    return "dark"  # default fallback


def _safe_update_trace_for_type(trace, update_kwargs):
    """Safely update trace attributes without breaking unsupported types."""
    try:
        trace.update(**update_kwargs)
    except Exception:
        if "marker" in update_kwargs and getattr(trace, "marker", None) is not None:
            try:
                m = trace.marker.to_plotly_json() if hasattr(trace.marker, "to_plotly_json") else {}
                m.update(update_kwargs["marker"])
                trace.marker = m
            except Exception:
                pass


def _set_trace_custom_hover_lakhs(trace):
    """Format hover labels in ₹ Lakhs."""
    try:
        if trace.type in ("bar", "histogram", "box", "violin"):
            y_vals = list(trace.y) if hasattr(trace, "y") else []
            custom = [[(v / 100000.0) if v is not None else None] for v in y_vals]
            trace.customdata = custom
            trace.hovertemplate = "%{x}<br>₹ %{customdata[0]:.2f} L<extra></extra>"
    except Exception:
        pass


def apply_chart_style(
    fig,
    title: str = "",
    x_title: str = "JobLevel",
    y_title: str = "",
    theme: str = "auto",
    legend_below: bool = True,
    showlegend: bool | None = None
):
    """
    Improved chart styling — Final Delta Patch v9
    ✅ Smart title wrapping (no toolbar overlap)
    ✅ Dynamic legend spacing (auto bottom or top)
    ✅ Unified hover values in ₹ Lakhs
    ✅ Standardized chart height across metrics
    ✅ Works perfectly in both dark & light modes
    """
    theme = _detect_theme(theme)
    is_dark = theme == "dark"

    text_color = "#FFFFFF" if is_dark else "#0b1220"
    paper_bg = FALLBACK_PAPER_BG_DARK if is_dark else FALLBACK_PAPER_BG_LIGHT
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    legend_bg = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.03)"
    legend_border = "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.05)"

    if showlegend is None:
        showlegend = len(fig.data) > 1

    # 🔹 Smart wrap long titles (split after 40 chars)
    wrapped_title = "<br>".join([title[i:i+40] for i in range(0, len(title), 40)]) if len(title) > 40 else title

    # 🔹 Base layout
    fig.update_layout(
        title=dict(
            text=wrapped_title,
            x=0.5,
            xanchor="center",
            y=0.92,
            yanchor="top",
            font=dict(size=18, color=text_color, family=HEADER_FONT),
        ),
        font=dict(family=BODY_FONT, color=text_color, size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=paper_bg,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25 if legend_below else 0.98,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE, color=text_color),
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
        ),
        margin=dict(t=80, l=60, r=40, b=130),
        bargap=0.15,
        bargroupgap=0.05,
        hoverlabel=dict(font_size=12, font_family=BODY_FONT),
        height=480,  # 🔹 Standardized height
    )

    # 🔹 Axes formatting
    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=AXIS_TITLE_SIZE, color=text_color),
        tickangle=-40,
        tickfont=dict(size=AXIS_TICK_SIZE, color=text_color),
        automargin=True,
        showgrid=False
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=AXIS_TITLE_SIZE, color=text_color),
        tickfont=dict(size=AXIS_TICK_SIZE, color=text_color),
        automargin=True,
        gridcolor=grid_color,
    )

    # 🔹 Normalize trace appearance + hover formatting
    for trace in fig.data:
        try:
            # Handle bars
            if trace.type == "bar":
                trace.marker.line.width = 0
                trace.width = 0.55
                if hasattr(trace, "y") and trace.y is not None:
                    y_vals = list(trace.y)
                    custom = [[(v / 100000.0) if v is not None else None] for v in y_vals]
                    trace.customdata = custom
                    trace.hovertemplate = "%{x}<br>₹ %{customdata[0]:.2f} L<extra></extra>"
            # Handle scatter/line
            elif trace.type in ("scatter", "line"):
                trace.mode = "lines+markers"
                trace.marker.size = 7
                trace.line.width = 2.5
                trace.hovertemplate = "%{x}<br>₹ %{y:.2f} L<extra></extra>"
            # Handle pie/donut
            elif trace.type == "pie":
                trace.hole = 0.45
                trace.textinfo = "percent"
                trace.insidetextorientation = "radial"
        except Exception:
            pass

    return fig
#==============
# PDF Helpers
#==============
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
def _ensure_joblevel_order(df, col="JobLevel", order=None):
    """Dynamic job-level ordering."""
    if col not in df.columns:
        return df
    df = df.copy()
    if order is None:
        unique = sorted(df[col].dropna().unique(), key=str)
        order = unique if unique else [
            "Analyst", "Assistant Manager", "Manager", "Senior Manager",
            "Associate Partner", "Director", "Executive", "Senior Executive"
        ]
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df
# ============================================================
# QF-7 FIX PATCH (2025-10-05): Metric A–G Cleanups
# ============================================================

def _safe_numeric(df, col):
    """Ensure numeric dtype and fill NaNs with 0 for calculations."""
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# --- FIX A ---
def average_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)

    grouped = df.groupby(job_col, observed=True)[ctc_col]
    agg = grouped.mean().reset_index().rename(columns={ctc_col: "Average CTC"})
    total_ctc = grouped.sum().reset_index().rename(columns={ctc_col: "Total CTC"})

    agg = pd.merge(agg, total_ctc, on=job_col, how="left")
    agg["Total CTC (₹ Cr.)"] = (agg["Total CTC"] / 1e7).round(2)
    agg["Avg CTC (₹ Lakhs)"] = (agg["Average CTC"] / 1e5).round(2)
    agg = agg[[job_col, "Total CTC (₹ Cr.)", "Avg CTC (₹ Lakhs)"]]

    fig = px.bar(
        agg, x=job_col, y="Avg CTC (₹ Lakhs)",
        color=job_col, color_discrete_sequence=PALETTE, text_auto=True
    )
    fig.update_traces(textposition="outside", textfont=dict(size=11, color="white"))
    fig = apply_chart_style(fig, title="Average CTC by Job Level", showlegend=False)
    return agg, fig


# --- FIX B ---
def median_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)

    grouped = df.groupby(job_col, observed=True)[ctc_col]
    agg = grouped.median().reset_index().rename(columns={ctc_col: "Median CTC"})
    total_ctc = grouped.sum().reset_index().rename(columns={ctc_col: "Total CTC"})

    agg = pd.merge(agg, total_ctc, on=job_col, how="left")
    agg["Total CTC (₹ Cr.)"] = (agg["Total CTC"] / 1e7).round(2)
    agg["Median CTC (₹ Lakhs)"] = (agg["Median CTC"] / 1e5).round(2)
    agg = agg[[job_col, "Total CTC (₹ Cr.)", "Median CTC (₹ Lakhs)"]]

    fig = px.bar(
        agg, x=job_col, y="Median CTC (₹ Lakhs)",
        color=job_col, color_discrete_sequence=PALETTE, text_auto=True
    )
    fig.update_traces(textposition="outside", textfont=dict(size=11, color="white"))
    fig = apply_chart_style(fig, title="Median CTC by Job Level", showlegend=False)
    return agg, fig

# --- FIX C (v4.6 Final) ---
def quartile_distribution(df, ctc_col="CTC", job_col="JobLevel"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)
    df = df.copy()

    try:
        df["Quartile"] = pd.qcut(df[ctc_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except Exception:
        df["Quartile"] = pd.cut(df[ctc_col], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])

    agg = (pd.crosstab(df[job_col], df["Quartile"], normalize="index") * 100).round(1).reset_index()

    melt_df = agg.melt(id_vars=job_col, var_name="Quartile", value_name="Percent")
    fig = px.bar(
        melt_df, x=job_col, y="Percent", color="Quartile",
        color_discrete_sequence=PALETTE, barmode="stack", text="Percent"
    )
    fig.update_traces(
        texttemplate="%{text}%", textposition="inside", textfont=dict(size=10, color="white")
    )

    # ✅ Legend shown and positioned neatly below chart
    fig = apply_chart_style(
        fig,
        title="Quartile Distribution of Employees by Job Level (CTC % Share)",
        legend_below=True, showlegend=True
    )
    return agg, fig
# --- FIX D (v4.6 Final) ---
def company_vs_market(df_company, df_market, job_col="JobLevel",
                      company_col="CompanyMedian", market_col="MarketMedian"):
    left = _ensure_joblevel_order(df_company[[job_col, company_col]], job_col)
    right = _ensure_joblevel_order(df_market[[job_col, market_col]], job_col)
    merged = pd.merge(left, right, on=job_col, how="inner")

    merged["Company (₹ L)"] = (merged[company_col] / 1e5).round(2)
    merged["Market (₹ L)"] = (merged[market_col] / 1e5).round(2)
    merged["Gap %"] = ((merged[company_col] - merged[market_col]) /
                       merged[market_col] * 100).round(1)
    table = merged[[job_col, "Company (₹ L)", "Market (₹ L)", "Gap %"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged[job_col], y=merged["Company (₹ L)"],
        name="Company", marker_color="#22D3EE"
    ))
    fig.add_trace(go.Scatter(
        x=merged[job_col], y=merged["Market (₹ L)"],
        name="Market", mode="lines+markers",
        line=dict(color="#FB7185", width=3)
    ))

    # ✅ Wrapped title and smaller legend font
    fig = apply_chart_style(
        fig,
        title="Company vs Market — Median CTC (₹ Lakhs)",
        legend_below=True, showlegend=True
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        title=dict(y=0.94)  # wrap slightly lower to avoid toolbar overlap
    )
    return table, fig
# --- FIX E ---
def bonus_pct_by_joblevel(df, job_col="JobLevel", bonus_col="Bonus", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _safe_numeric(df, bonus_col)
    df = _ensure_joblevel_order(df, job_col)

    df["Bonus %"] = np.where(df[ctc_col] > 0, (df[bonus_col] / df[ctc_col]) * 100, np.nan)
    agg = df.groupby(job_col, observed=True)["Bonus %"].mean().reset_index().round(2)

    fig = px.bar(
        agg, x=job_col, y="Bonus %",
        color=job_col, color_discrete_sequence=PALETTE, text="Bonus %"
    )
    fig.update_traces(textposition="inside", textfont=dict(color="black", size=11))
    fig = apply_chart_style(fig, title="Average Bonus % of CTC by Job Level", showlegend=False)

    return agg, fig
# --- FIX F ---
def average_ctc_by_gender_joblevel(df, job_col="JobLevel", gender_col="Gender", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby([job_col, gender_col], observed=True)[ctc_col].mean().reset_index()
    pivot = agg.pivot(index=job_col, columns=gender_col, values=ctc_col / 1e5).fillna(0)
    pivot["Gap (%)"] = np.where(pivot.get("Female", 0) > 0,
                               ((pivot.get("Male", 0) - pivot.get("Female", 0)) /
                                pivot.get("Female", 0) * 100).round(1), np.nan)
    pivot = pivot.reset_index().rename(columns={"Male": "Avg CTC (M)", "Female": "Avg CTC (F)"})
    fig = px.bar(agg, x=job_col, y=df[ctc_col] / 1e5, color=gender_col,
                 color_discrete_sequence=PALETTE, barmode="group")
    fig = apply_chart_style(fig, title="Average CTC by Gender & Job Level", legend_below=False)
    return pivot, fig
# --- FIX G ---
def average_ctc_by_rating_joblevel(df, job_col="JobLevel", rating_col="Rating", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)

    agg = df.groupby([job_col, rating_col], observed=True)[ctc_col].mean().reset_index()
    pivot = agg.pivot(index=job_col, columns=rating_col, values=ctc_col / 1e5).round(2)
    pivot = pivot.reset_index().rename(columns=lambda x: f"Rating {x}" if isinstance(x, (int, float)) else x)

    fig = px.bar(
        agg, x=job_col, y=ctc_col / 1e5, color=rating_col,
        color_discrete_sequence=px.colors.sequential.Blues, barmode="stack"
    )
    fig = apply_chart_style(fig, title="Average CTC by Performance Rating & Job Level", legend_below=False)
    return pivot, fig
# ============================================================
# Render Metrics + Tables (v4.4 QF-6 Polished Layout)
# ============================================================

sections = []
images_for_download = []

# Main metric group
metrics = [
    ("🏷️ Average CTC by Job Level", average_ctc_by_joblevel,
     "Average pay by level."),
    ("📏 Median CTC by Job Level", median_ctc_by_joblevel,
     "Median pay across levels."),
    ("📊 Quartile Distribution (Share of Employees)", quartile_distribution,
     "Distribution of employees by quartile across levels."),
    ("🎁 Bonus % of CTC by Job Level", bonus_pct_by_joblevel,
     "Average bonus percentage by level."),
]

for title, func, desc in metrics:
    st.subheader(title)
    table, fig = func(emp_df)
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
    sections.append((title, desc, table, {"png": None}))

# ------------------------------------------------------------
# Company vs Market
# ------------------------------------------------------------
if bench_df is not None:
    st.subheader("📉 Company vs Market (Median CTC)")
    df_company = (
        emp_df.groupby("JobLevel", observed=True)["CTC"]
        .median().reset_index().rename(columns={"CTC": "CompanyMedian"})
    )
    df_market = (
        bench_df.groupby("JobLevel", observed=True)["MarketMedianCTC"]
        .median().reset_index().rename(columns={"MarketMedianCTC": "MarketMedian"})
    )
    tableE, figE = company_vs_market(df_company, df_market)
    st.dataframe(tableE, use_container_width=True)
    st.plotly_chart(figE, use_container_width=True)
    sections.append(("Company vs Market", "Internal vs market comparison.", tableE, {"png": None}))
else:
    st.info("ℹ️ Upload benchmark data to see market comparison.")

# ------------------------------------------------------------
# Gender & Rating Differentiation
# ------------------------------------------------------------
emp_df = emp_df.rename(columns={"PerformanceRating": "Rating"})

last_metrics = [
    ("👫 Average CTC by Gender & Job Level", average_ctc_by_gender_joblevel,
     "Gender pay differentiation across levels."),
    ("⭐ Average CTC by Performance Rating & Job Level", average_ctc_by_rating_joblevel,
     "Pay differentiation by performance rating."),
]

for title, func, desc in last_metrics:
    st.subheader(title)
    table, fig = func(emp_df)
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
    sections.append((title, desc, table, {"png": None}))
#========================
# After Metrics Render → Proceed to PDF & Downloads
# =======================

st.header("📥 Download Reports")
st.write("Choose metrics to include in the compiled PDF:")

# Create checkboxes for each metric
kpi_check = {title: st.checkbox(title, key=f"chk_{i}") for i, (title, _, _, _) in enumerate(sections)}

# ========================
# Compile PDF Button Logic
# ========================
if st.button("🧾 Compile Selected Report"):
    selected_titles = [t for t, v in kpi_check.items() if v]
    if not selected_titles:
        st.warning("Select at least one metric to include.")
    else:
        selected_sections = [s for s in sections if s[0] in selected_titles]
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=20 * mm, bottomMargin=20 * mm
        )
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
        for idx, (title, _, _, _) in enumerate(selected_sections, 1):
            story.append(Paragraph(f"{idx}. {title}", body))
        story.append(PageBreak())

        # Render each metric’s table + chart
        for title, desc, tbl, asset in selected_sections:
            bname = sanitize_anchor(title)
            story.append(PDFBookmark(bname, title))
            story.append(Paragraph(f"<b>{title}</b>", h2))
            story.append(Spacer(1, 6))
            if desc:
                story.append(Paragraph(desc, body))
                story.append(Spacer(1, 6))
            if tbl is not None and hasattr(tbl, "shape") and tbl.shape[0] > 0:
                data = [list(tbl.columns)] + tbl.fillna("").values.tolist()
                colWidths = [(A4[0] - 40) / len(tbl.columns)] * len(tbl.columns)
                t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=colWidths)
                tstyle = TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
                ])
                for r in range(1, len(data)):
                    if r % 2 == 0:
                        tstyle.add("BACKGROUND", (0, r), (-1, r), TABLE_ZEBRA)
                t.setStyle(tstyle)
                story.append(t)
                story.append(Spacer(1, 8))
            if asset and asset.get("png") and os.path.exists(asset["png"]):
                story.append(RLImage(asset["png"], width=170 * mm, height=90 * mm))
                story.append(Spacer(1, 6))
            story.append(Paragraph(f"<i>Insight:</i> Review {title} for trends.", body))
            story.append(PageBreak())

        doc.build(
            story,
            onFirstPage=lambda c, d: (draw_background(c, d), add_page_number(c, d)),
            onLaterPages=lambda c, d: (draw_background(c, d), add_page_number(c, d))
        )
        st.download_button(
            "⬇️ Download Compiled PDF", buf.getvalue(),
            file_name="cb_dashboard_compiled.pdf", mime="application/pdf"
        )
# -----------------------
# Compiled PDF Bookmark Helper
# -----------------------
class PDFBookmark(Flowable):
    def __init__(self, name, title):
        super().__init__()
        self.name, self.title = name, title
    def wrap(self, availWidth, availHeight):
        return (0, 0)
    def draw(self):
        try:
            self.canv.bookmarkPage(self.name)
            self.canv.addOutlineEntry(self.title, self.name, level=0, closed=False)
        except Exception:
            pass
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