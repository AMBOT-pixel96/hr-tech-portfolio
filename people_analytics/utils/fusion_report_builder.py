# ============================================
# utils/fusion_report_builder.py — v2.3 | Unicode + Layout Polish
# ============================================
"""
Generates the Fusion Insights Report:
- Auto-detects cross-module correlations
- Embeds visual plots (Plotly → PNG)
- Summarizes Chatbot memory
- Uses full Unicode font for emojis & bullets
"""

import io, os
from datetime import datetime
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.express as px

# --------------------------------------------
# Font registration (Unicode-safe)
# --------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# --------------------------------------------
# Styles
# --------------------------------------------
def _get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName=DEFAULT_FONT, fontSize=22, alignment=1,
                                textColor=colors.HexColor("#0F172A")),
        "heading": ParagraphStyle("heading", parent=base["Heading2"],
                                  fontName=DEFAULT_FONT, fontSize=13,
                                  textColor=colors.HexColor("#1E3A8A"), spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontName=DEFAULT_FONT, fontSize=10,
                               leading=13, textColor=colors.black),
    }

# --------------------------------------------
# Detect cross-module fusions
# --------------------------------------------
def detect_fusions(modules):
    fusions = []
    try:
        # Engagement ↔ Attrition
        if "attrition" in modules and "engagement" in modules:
            a, e = modules["attrition"], modules["engagement"]
            if {"Department", "AttritionFlag"}.issubset(a.columns) and "EngagementIndex" in e.columns:
                m = a.merge(e, on="Department", suffixes=("_attr", "_eng"))
                m["AttritionBinary"] = m["AttritionFlag"].map({"Yes": 1, "No": 0})
                corr = m["AttritionBinary"].corr(m["EngagementIndex"])
                fig = px.scatter(m, x="EngagementIndex", y="AttritionBinary",
                                 trendline="ols", title="Engagement vs Attrition")
                fusions.append(("Engagement ↔ Attrition", f"Correlation: {corr:.2f}", fig))

        # Performance ↔ Compensation
        if "compensation" in modules and "performance" in modules:
            c, p = modules["compensation"], modules["performance"]
            if {"EmployeeID", "CTC"}.issubset(c.columns) and {"EmployeeID", "PerformanceRating"}.issubset(p.columns):
                m = c.merge(p, on="EmployeeID", how="inner")
                corr = m["CTC"].corr(m["PerformanceRating"])
                fig = px.scatter(m, x="PerformanceRating", y="CTC",
                                 trendline="ols", title="Compensation vs Performance")
                fusions.append(("Performance ↔ Compensation", f"Correlation: {corr:.2f}", fig))

        # Gender Pay Gap
        if "compensation" in modules:
            c = modules["compensation"]
            if {"Gender", "CTC"}.issubset(c.columns):
                g = c.groupby("Gender")["CTC"].mean().round(0)
                gap = (g.max() - g.min()) / g.max() * 100
                fig = px.bar(g, x=g.index, y=g.values, title="Average CTC by Gender")
                fusions.append(("Gender Pay Gap", f"Gap: {gap:.1f}% — {g.idxmax()}s earn more", fig))

        return fusions
    except Exception as e:
        return [("Fusion Analysis Failed", str(e), None)]

# --------------------------------------------
# Build Fusion Insights PDF
# --------------------------------------------
def build_fusion_report(modules, messages):
    buf = io.BytesIO()
    s = _get_styles()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm
    )
    story = []

    # Cover Page
    story.append(Spacer(1, 100))
    story.append(Paragraph("📘 Fusion Insights Report", s["title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y')}</font></para>",
        s["body"]
    ))
    story.append(PageBreak())

    # Fusion Analysis Section
    story.append(Paragraph("🔮 Automated Cross-Module Insights", s["heading"]))
    story.append(Spacer(1, 6))

    fusions = detect_fusions(modules)
    for title, summary, fig in fusions:
        story.append(Paragraph(f"<b>{title}</b>", s["heading"]))
        story.append(Paragraph(summary, s["body"]))
        story.append(Spacer(1, 6))

        if fig:
            img_path = f"/tmp/{title.replace(' ', '_')}_{int(datetime.now().timestamp())}.png"
            try:
                fig.write_image(img_path, width=800, height=500)
                story.append(RLImage(img_path, width=160 * mm, height=90 * mm))
                os.remove(img_path)  # cleanup
            except Exception:
                pass
            story.append(Spacer(1, 15))
        story.append(PageBreak())

    # Chatbot Memory Section
    story.append(Paragraph("💬 Chatbot Memory Summary", s["heading"]))
    story.append(Spacer(1, 4))
    chat = messages if messages else [{"role": "system", "content": "No chatbot history found."}]
    for msg in chat:
        prefix = "👤 " if msg.get("role") == "user" else "🤖 "
        story.append(Paragraph(f"{prefix}{msg.get('content', '')}", s["body"]))
        story.append(Spacer(1, 5))
        story.append(Paragraph("<font color='#9CA3AF'>───────────────────────</font>", s["body"]))
        story.append(Spacer(1, 5))

    doc.build(story)
    return buf.getvalue()