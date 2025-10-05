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
    ⚠️ Session resets after idle. Download PDF to save results.
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
# Helpers (v6)
# -----------------------
import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# Optional for PDF/report helpers (if you're using reportlab)
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
except Exception:
    rl_colors = None
    A4 = (595.27, 841.89)

# Temp dir to save images/html if needed (change if you have a TMP_DIR)
TMP_DIR = os.environ.get("TMP_DIR", "/tmp")

# -----------------------
# Visual / PDF constants
# -----------------------
# A palette that works well on both light & dark backgrounds (plotly qualitative).
PALETTE = px.colors.qualitative.Plotly

# Fonts & sizes
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TITLE_SIZE = 20
AXIS_TITLE_SIZE = 12
AXIS_TICK_SIZE = 11
LEGEND_FONT_SIZE = 12

# A neutral fallback background that looks good in both themes if needed
FALLBACK_PAPER_BG_DARK = "#0b1220"   # deep navy / almost black
FALLBACK_PAPER_BG_LIGHT = "#FFFFFF"

# -----------------------
# Basic helpers
# -----------------------
def validate_exact_headers(df_or_cols, required_cols):
    """Return (bool, msg). Exact order & names expected."""
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    ok = cols == required_cols
    return (ok, "OK" if ok else f"Header mismatch. Expected {required_cols}, found {cols}")

def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def safe_filename(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def readable_lakhs_number(x):
    """Return value in Lakhs (float) or None."""
    if pd.isna(x):
        return None
    try:
        return round(float(x) / 100000.0, 2)
    except Exception:
        return None

# -----------------------
# PDF helpers (optional)
# -----------------------
def draw_background(canvas, doc):
    if rl_colors is None:
        return
    canvas.saveState()
    canvas.setStrokeColor(rl_colors.black)
    canvas.rect(5, 5, A4[0]-10, A4[1]-10, stroke=1, fill=0)
    canvas.restoreState()

def add_page_number(canvas, doc):
    if rl_colors is None:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

def save_plotly_asset(fig, filename_base, width=1200, height=700, scale=2):
    """
    Save plotly figure as PNG; fallback to HTML if PNG generation fails.
    Returns dict {"png": path or None, "html": path or None}
    """
    base = os.path.join(TMP_DIR, filename_base)
    png_path, html_path = base + ".png", base + ".html"
    try:
        # Try to render PNG first — strip trace outline for cleaner images
        for t in fig.data:
            try:
                if getattr(t, "marker", None) is not None:
                    t.marker.line = dict(width=0)
            except Exception:
                pass
        fig.update_layout(template="plotly_white", title_font=dict(size=TITLE_SIZE, color="black", family=BODY_FONT))
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        return {"png": png_path, "html": None}
    except Exception:
        try:
            fig.write_html(html_path)
            return {"png": None, "html": html_path}
        except Exception:
            return {"png": None, "html": None}

# -----------------------
# Theme detection & style applier
# -----------------------
def _detect_theme(theme_arg="auto"):
    """
    Return 'dark' or 'light'.
    If theme_arg is 'auto', try to use streamlit theme setting (st.get_option).
    """
    if theme_arg in ("dark", "light"):
        return theme_arg
    try:
        base = st.get_option("theme.base")
        if base and base.lower() in ("dark", "light"):
            return base.lower()
    except Exception:
        pass
    # fallback: prefer dark since your screenshots are night-mode
    return "dark"

def _safe_update_trace_for_type(trace, update_kwargs):
    """Update a single trace but don't raise if properties unsupported."""
    try:
        trace.update(**update_kwargs)
    except Exception:
        # try individually for marker line because different trace types have different attrs
        try:
            if "marker" in update_kwargs and getattr(trace, "marker", None) is not None:
                m = trace.marker.to_plotly_json() if hasattr(trace.marker, "to_plotly_json") else {}
                new_marker = update_kwargs["marker"]
                # merge gently
                m.update(new_marker)
                trace.marker = m
        except Exception:
            pass

def _set_trace_custom_hover_lakhs(trace):
    """
    For bar/column-like traces we add customdata with lakhs and set a concise hovertemplate.
    Works safely across trace types.
    """
    try:
        if trace.type in ("bar", "histogram", "box", "violin"):
            # trace.y might be a tuple/list/np.ndarray
            y_vals = list(trace.y) if hasattr(trace, "y") else []
            custom = [[(v / 100000.0) if v is not None else None] for v in y_vals]
            trace.customdata = custom
            # show lakhs with 2 decimals
            trace.hovertemplate = "%{x}<br>₹ %{customdata[0]:.2f} L<extra></extra>"
    except Exception:
        pass

def apply_chart_style(fig, title: str = "", x_title: str = "JobLevel", y_title: str = "", theme: str = "auto", legend_below: bool = True, showlegend: bool | None = None):
    """
    Apply consistent styling to a Plotly figure (safe to call on any figure).
    - Centers title (avoids title overflow)
    - Puts legend below chart (horizontal) and adds bottom margin to avoid overlap
    - Uses automargin on xaxis/yaxis and sets tick angles for better mobile layout
    - Formats hover labels & optionally adds lakhs hover for bar traces
    - Safely updates traces (no global fig.update_traces(...) that would crash on pie/scatter)
    """
    theme = _detect_theme(theme)
    is_dark = (theme == "dark")

    # colours based on theme
    text_color = "#FFFFFF" if is_dark else "#0b1220"
    paper_bg = FALLBACK_PAPER_BG_DARK if is_dark else FALLBACK_PAPER_BG_LIGHT
    plot_bg = "rgba(0,0,0,0)"  # let the page/card BG show through
    grid_color = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.08)"
    legend_bg = "rgba(255,255,255,0.02)" if is_dark else "rgba(0,0,0,0.03)"
    legend_border = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.06)"

    # Decide whether to show legend when user didn't pass explicit param:
    if showlegend is None:
        # If the figure has >1 distinct trace and not all traces are single color identical to x, show legend
        showlegend = len(fig.data) > 1

    # Layout base
    bottom_margin = 140 if legend_below and showlegend else 80
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top", font=dict(size=TITLE_SIZE, color=text_color, family=HEADER_FONT)),
        title_x=0.5,
        font=dict(family=BODY_FONT, color=text_color, size=12),
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28 if legend_below else 0.98,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE, color=text_color),
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
        ),
        margin=dict(t=90, l=70, r=40, b=bottom_margin),
        bargap=0.12,         # thicker bars
        bargroupgap=0.02,
        hoverlabel=dict(font_size=12, font_family=BODY_FONT),
    )

    # Axis settings (automargins and tick orientation)
    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=AXIS_TITLE_SIZE, color=text_color, family=BODY_FONT),
        tickangle=-45,
        tickfont=dict(size=AXIS_TICK_SIZE, color=text_color),
        automargin=True,
        showgrid=False
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=AXIS_TITLE_SIZE, color=text_color, family=BODY_FONT),
        tickfont=dict(size=AXIS_TICK_SIZE, color=text_color),
        automargin=True,
        gridcolor=grid_color,
    )

    # Remove legend title to avoid extra vertical text that overlaps
    try:
        fig.update_layout(legend_title_text="")
    except Exception:
        pass

    # Safe per-trace updates: marker lines, hover templates for bars, ensure bar width only applied where valid
    for trace in fig.data:
        try:
            if trace.type in ("bar", "histogram"):
                # thicker bars, no outline
                _safe_update_trace_for_type(trace, {"marker": {"line": {"width": 0}}})
                _set_trace_custom_hover_lakhs(trace)
                # set width *only* if trace supports it
                try:
                    _safe_update_trace_for_type(trace, {"width": 0.6})
                except Exception:
                    pass
            elif trace.type in ("scatter", "line"):
                # nice markers for lines in company vs market
                _safe_update_trace_for_type(trace, {"mode": "lines+markers", "marker": {"size": 8, "line": {"width": 1}}})
            elif trace.type == "pie":
                # donut style with inside percent labels
                try:
                    _safe_update_trace_for_type(trace, {"hole": 0.45, "textinfo": "percent", "insidetextorientation": "radial"})
                except Exception:
                    pass
            else:
                # generic try for other trace types
                try:
                    _safe_update_trace_for_type(trace, {"marker": {"line": {"width": 0}}})
                except Exception:
                    pass
        except Exception:
            # swallow to avoid interrupting app
            pass

    # done
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
# Metrics block (Final Stable v6)
# -----------------------
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_JOBLEVEL_ORDER = [
    "Analyst",
    "Assistant Manager",
    "Associate Partner",
    "Director",
    "Executive",
    "Manager",
    "Senior Executive",
    "Senior Manager",
]

def _ensure_joblevel_order(df, col="JobLevel", order=DEFAULT_JOBLEVEL_ORDER):
    """If JobLevel present, make it categorical with desired order for consistent plotting."""
    if col in df.columns:
        df = df.copy()
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


# -----------------------
# 1️⃣ Average CTC by Job Level
# -----------------------
def average_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby(job_col, observed=True)[ctc_col].mean().reset_index()
    agg["ctc_lakhs"] = agg[ctc_col] / 100000.0

    fig = px.bar(
        agg,
        x=job_col,
        y="ctc_lakhs",
        color=job_col,
        color_discrete_sequence=PALETTE,
        labels={"ctc_lakhs": "Avg CTC (₹ Lakhs)"},
    )
    fig.update_layout(showlegend=False)
    fig = apply_chart_style(fig, title="Average CTC by Job Level")
    return fig


# -----------------------
# 2️⃣ Median CTC by Job Level
# -----------------------
def median_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby(job_col, observed=True)[ctc_col].median().reset_index()
    agg["ctc_lakhs"] = agg[ctc_col] / 100000.0

    fig = px.bar(
        agg,
        x=job_col,
        y="ctc_lakhs",
        color=job_col,
        color_discrete_sequence=PALETTE,
        labels={"ctc_lakhs": "Median CTC (₹ Lakhs)"},
    )
    fig.update_layout(showlegend=False)
    fig = apply_chart_style(fig, title="Median CTC by Job Level")
    return fig


# -----------------------
# 3️⃣ Quartile Distribution (Share of Employees)
# -----------------------
def quartile_distribution(df, job_col="JobLevel", ctc_col="CTC"):
    if "Quartile" in df.columns:
        qdf = df.groupby("Quartile").size().reset_index(name="count")
    else:
        qlabels = ["Q1", "Q2", "Q3", "Q4"]
        df = df.copy()
        df["Quartile"] = pd.qcut(df[ctc_col], q=4, labels=qlabels)
        qdf = df.groupby("Quartile").size().reset_index(name="count")

    fig = px.pie(qdf, names="Quartile", values="count", color_discrete_sequence=PALETTE)
    fig.update_traces(hole=0.45, textinfo="percent+label")
    fig = apply_chart_style(fig, title="Quartile Distribution (Share of Employees)")
    return fig


# -----------------------
# 4️⃣ Average CTC by Gender & Job Level
# -----------------------
def average_ctc_by_gender_joblevel(df, job_col="JobLevel", gender_col="Gender", ctc_col="CTC"):
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby([job_col, gender_col], observed=True)[ctc_col].mean().reset_index()
    agg["ctc_lakhs"] = agg[ctc_col] / 100000.0

    fig = px.bar(
        agg,
        x=job_col,
        y="ctc_lakhs",
        color=gender_col,
        barmode="group",
        color_discrete_sequence=PALETTE,
        labels={"ctc_lakhs": "Avg CTC (₹ Lakhs)"},
    )
    fig = apply_chart_style(fig, title="Average CTC by Gender & Job Level")
    return fig


# -----------------------
# 5️⃣ Average CTC by Performance Rating & Job Level
# -----------------------
def average_ctc_by_rating_joblevel(df, job_col="JobLevel", rating_col="Rating", ctc_col="CTC"):
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby([job_col, rating_col], observed=True)[ctc_col].mean().reset_index()
    agg["ctc_lakhs"] = agg[ctc_col] / 100000.0

    fig = px.bar(
        agg,
        x=job_col,
        y="ctc_lakhs",
        color=rating_col,
        barmode="group",
        color_discrete_sequence=PALETTE,
        labels={"ctc_lakhs": "Avg CTC (₹ Lakhs)", rating_col: "Rating"},
    )
    fig = apply_chart_style(fig, title="Average CTC by Performance Rating & Job Level")
    return fig
# -----------------------
# 6️⃣ Company vs Market (Median CTC)
# -----------------------
def company_vs_market(df_company, df_market, job_col="JobLevel", company_col="CompanyMedian", market_col="MarketMedian"):
    """
    Combined bar (Company) + line (Market) comparison by job level.
    df_company and df_market should share same JobLevel column.
    """
    left = _ensure_joblevel_order(df_company[[job_col, company_col]].copy(), job_col)
    right = _ensure_joblevel_order(df_market[[job_col, market_col]].copy(), job_col)
    merged = pd.merge(left, right, on=job_col, how="inner")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged[job_col],
        y=merged[company_col],
        name="Company",
        marker_color="#22D3EE",  # teal accent
        opacity=0.9
    ))
    fig.add_trace(go.Scatter(
        x=merged[job_col],
        y=merged[market_col],
        name="Market",
        mode="lines+markers",
        line=dict(color="#FB7185", width=3),
        marker=dict(size=7)
    ))

    fig = apply_chart_style(fig, title="Company vs Market — Median CTC (₹ Lakhs)")
    return fig


# -----------------------
# 7️⃣ Bonus % of CTC by Job Level
# -----------------------
def bonus_pct_by_joblevel(df, job_col="JobLevel", bonus_col="Bonus", ctc_col="CTC"):
    """
    Bar chart: Average Bonus % of CTC by Job Level.
    Displays neat pastel bars, white title, and centered legend.
    """
    df = _ensure_joblevel_order(df, job_col)
    df = df.copy()
    df["Bonus %"] = np.where(df[ctc_col] > 0, (df[bonus_col] / df[ctc_col]) * 100, np.nan)

    agg = df.groupby(job_col, observed=True)["Bonus %"].mean().reset_index()
    agg["Bonus %"] = agg["Bonus %"].round(2)

    fig = px.bar(
        agg,
        x=job_col,
        y="Bonus %",
        color=job_col,
        color_discrete_sequence=PALETTE,
        labels={"Bonus %": "Avg Bonus (%)"},
    )

    fig = apply_chart_style(fig, title="Average Bonus % of CTC by Job Level")
    fig.update_layout(showlegend=False)
    return fig
# -----------------------
# Initialize metric storage
# -----------------------
sections = []      # stores metric titles, tables, descriptions, and assets
images_for_download = []   # stores images for quick chart downloads
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
# -----------------------
# Render All Metrics (Dashboard Display + PDF Linking)
# -----------------------

# 1️⃣ Average CTC by Job Level
st.subheader("🏷️ Average CTC by Job Level")
figA = average_ctc_by_joblevel(emp_df)
st.plotly_chart(figA, use_container_width=True)
sections.append(("Average CTC by Job Level", "Average pay by level.", None, {"png": None}))

# 2️⃣ Median CTC by Job Level
st.subheader("📏 Median CTC by Job Level")
figB = median_ctc_by_joblevel(emp_df)
st.plotly_chart(figB, use_container_width=True)
sections.append(("Median CTC by Job Level", "Median pay across job levels.", None, {"png": None}))

# 3️⃣ Quartile Distribution (Share of Employees)
st.subheader("📉 Quartile Distribution (Share of Employees)")
figC = quartile_distribution(emp_df)
st.plotly_chart(figC, use_container_width=True)
sections.append(("Quartile Distribution (Share of Employees)", "Proportion of employees across pay quartiles.", None, {"png": None}))

# 4️⃣ Bonus % of CTC by Job Level
st.subheader("🎁 Bonus % of CTC by Job Level")
figD = bonus_pct_by_joblevel(emp_df)
st.plotly_chart(figD, use_container_width=True)
sections.append(("Bonus % of CTC by Job Level", "Average bonus share of CTC by level.", None, {"png": None}))

# 5️⃣ Company vs Market (Median CTC)
if bench_df is not None:
    st.subheader("📊 Company vs Market (Median CTC)")
    # Prepare company + market dataframes
    df_company = emp_df.groupby("JobLevel", observed=True)["CTC"].median().reset_index().rename(columns={"CTC": "CompanyMedian"})
    df_market = bench_df.groupby("JobLevel", observed=True)["MarketMedianCTC"].median().reset_index().rename(columns={"MarketMedianCTC": "MarketMedian"})
    figE = company_vs_market(df_company, df_market)
    st.plotly_chart(figE, use_container_width=True)
    sections.append(("Company vs Market (Median CTC)", "Comparison of internal vs market median pay levels.", None, {"png": None}))
else:
    st.info("ℹ️ Upload a benchmark dataset to view Company vs Market comparison.")

# 6️⃣ Average CTC by Gender & Job Level
st.subheader("👫 Average CTC by Gender & Job Level")
figF = average_ctc_by_gender_joblevel(emp_df)
st.plotly_chart(figF, use_container_width=True)
sections.append(("Average CTC by Gender & Job Level", "Gender pay differentiation across levels.", None, {"png": None}))

# 7️⃣ Average CTC by Performance Rating & Job Level
st.subheader("⭐ Average CTC by Performance Rating & Job Level")
# rename PerformanceRating → Rating to match function signature
emp_df = emp_df.rename(columns={"PerformanceRating": "Rating"})
figG = average_ctc_by_rating_joblevel(emp_df)
st.plotly_chart(figG, use_container_width=True)
sections.append(("Average CTC by Performance Rating & Job Level", "Pay differentiation by performance rating.", None, {"png": None}))
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