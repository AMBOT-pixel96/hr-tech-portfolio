import io, os
from datetime import datetime
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
import plotly.express as px

def _get_styles():
    base = getSampleStyleSheet()
    return {
        "heading": ParagraphStyle("heading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=13, textColor=colors.black),
    }

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
                fig = px.scatter(m, x="EngagementIndex", y="AttritionBinary", trendline="ols")
                fusions.append(("Engagement ↔ Attrition", f"Correlation: {corr:.2f}", fig))

        # Performance ↔ Compensation
        if "compensation" in modules and "performance" in modules:
            c, p = modules["compensation"], modules["performance"]
            if {"EmployeeID", "CTC"}.issubset(c.columns) and {"EmployeeID", "PerformanceRating"}.issubset(p.columns):
                m = c.merge(p, on="EmployeeID", how="inner")
                corr = m["CTC"].corr(m["PerformanceRating"])
                fig = px.scatter(m, x="PerformanceRating", y="CTC", trendline="ols")
                fusions.append(("Performance ↔ Compensation", f"Correlation: {corr:.2f}", fig))

        # Gender Pay Gap
        if "compensation" in modules:
            c = modules["compensation"]
            if {"Gender", "CTC"}.issubset(c.columns):
                g = c.groupby("Gender")["CTC"].mean().round(0)
                gap = (g.max() - g.min()) / g.max() * 100
                fig = px.bar(g, x=g.index, y=g.values)
                fusions.append(("Gender Pay Gap", f"Gap: {gap:.1f}% — {g.idxmax()}s earn more", fig))

        return fusions
    except Exception as e:
        return [("Fusion Analysis Failed", str(e), None)]

def build_fusion_report(modules, messages):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
    s = _get_styles()
    story = []

    # Cover
    story.append(Spacer(1, 100))
    story.append(Paragraph("<para align=center><font size=22><b>Fusion Insights Report</b></font></para>", s["body"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y')}</font></para>", s["body"]))
    story.append(PageBreak())

    # Fusion analysis
    story.append(Paragraph("🔮 Automated Cross-Module Insights", s["heading"]))
    story.append(Spacer(1, 6))
    fusions = detect_fusions(modules)

    for title, summary, fig in fusions:
        story.append(Paragraph(f"<b>{title}</b>", s["heading"]))
        story.append(Paragraph(summary, s["body"]))
        story.append(Spacer(1, 6))
        if fig:
            img_path = f"/tmp/{title.replace(' ', '_')}.png"
            fig.write_image(img_path, width=800, height=500)
            story.append(RLImage(img_path, width=160 * mm, height=90 * mm))
            story.append(Spacer(1, 15))
        story.append(PageBreak())

    # Chatbot memory
    story.append(Paragraph("💬 Chatbot Memory Summary", s["heading"]))
    chat = messages if messages else [{"role": "system", "content": "No chatbot history found."}]
    for msg in chat:
        prefix = "👤 " if msg["role"] == "user" else "🤖 "
        story.append(Paragraph(f"{prefix}{msg['content']}", s["body"]))
        story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()