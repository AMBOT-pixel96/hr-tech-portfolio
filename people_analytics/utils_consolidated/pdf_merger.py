# ============================================
# utils_consolidated/pdf_merger.py — v7.3 | Executive Aesthetic Build
# ============================================
import os
import io
import tempfile
import json
from datetime import datetime
from typing import List, Dict

# PDF libs
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    # prefer pypdf if available (modern), fallback to PyPDF2
    from pypdf import PdfReader, PdfWriter
except Exception:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except Exception:
        raise ImportError("pypdf / PyPDF2 required. Please install pypdf or PyPDF2 in the environment.")

# Register a unicode-capable font for robust rendering (₹ etc.)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# TMP_DIR expected to exist in package; fallback to /tmp/people_analytics_consolidated
TMP_DIR = os.path.join(tempfile.gettempdir(), "people_analytics_consolidated")
os.makedirs(TMP_DIR, exist_ok=True)

# Modules expected order
MODULES_ORDER = ["Attrition", "Compensation", "Engagement", "Performance", "Workforce"]

# -------------------------
# Helpers
# -------------------------
def _fmt_date_short(ts_path: str) -> str:
    """Return file mtime formatted as MMM'YY (no time)."""
    try:
        t = os.path.getmtime(ts_path)
        return datetime.fromtimestamp(t).strftime("%b'%y")
    except Exception:
        return "—"

def _load_module_meta(tmp_dir: str, module: str) -> Dict:
    """Load {module}.json metadata if exists, else return default."""
    jpath = os.path.join(tmp_dir, f"{module}.json")
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {"insights": "No summary provided.", "metrics_short": ""}
    return {"insights": "No summary provided.", "metrics_short": ""}

def _normalize_insights(ins):
    """Turn various forms into a list of strings."""
    if not ins:
        return []
    if isinstance(ins, list):
        return [str(x).strip() for x in ins if str(x).strip()]
    if isinstance(ins, str):
        # attempt to split if it's a long joined string separated by '•' or newline or ';'
        if "•" in ins:
            parts = [p.strip() for p in ins.split("•") if p.strip()]
            return parts
        if "\n" in ins:
            parts = [p.strip() for p in ins.splitlines() if p.strip()]
            return parts
        if ";" in ins:
            parts = [p.strip() for p in ins.split(";") if p.strip()]
            return parts
        return [ins.strip()]
    try:
        return [str(ins)]
    except Exception:
        return []

def _filter_insights_for_module(module: str, insights_list: List[str]) -> List[str]:
    """Apply module-specific filters to remove unwanted insights from consolidated executive summary."""
    filtered = []
    for s in insights_list:
        s_low = s.lower()
        drop = False
        if module.lower() == "attrition":
            # remove job level attrition & exit reasons lines
            keywords = ["job level", "job-level", "joblevel", "exit reason", "exit reasons"]
            if any(k in s_low for k in keywords):
                drop = True
        elif module.lower() == "compensation":
            keywords = ["gender pay gap", "gender pay", "internal vs market", "internal vs. market", "internal vs market"]
            if any(k in s_low for k in keywords):
                drop = True
        elif module.lower() == "engagement":
            keywords = ["demographic engagement", "demographic"]
            if any(k in s_low for k in keywords):
                drop = True
        elif module.lower() == "performance":
            # remove the "no chart available" cosmetic message if present
            if "no chart available" in s_low or "⚠️ no chart available" in s_low:
                drop = True
        elif module.lower() == "workforce":
            keywords = ["skill inventory", "skill", "skills inventory"]
            if any(k in s_low for k in keywords):
                drop = True
        if not drop:
            filtered.append(s)
    return filtered

# -------------------------
# ReportLab page builders
# -------------------------
PAGE_WIDTH, PAGE_HEIGHT = A4
styles = getSampleStyleSheet()
body_style = ParagraphStyle(
    "body",
    parent=styles["Normal"],
    fontName=DEFAULT_FONT,
    fontSize=10,
    leading=14,
    textColor=colors.black
)
heading_style = ParagraphStyle(
    "heading",
    parent=styles["Heading1"],
    fontName=DEFAULT_FONT,
    fontSize=18,
    leading=22,
    spaceAfter=8
)
subheading_style = ParagraphStyle(
    "subheading",
    parent=styles["Heading2"],
    fontName=DEFAULT_FONT,
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#1E3A8A"),
    spaceAfter=6
)

def _build_cover_pdf(path_out: str, title: str = "People Analytics Leadership Deck", subtitle: str = "Unified HR Insights | Executive Summary"):
    doc = SimpleDocTemplate(path_out, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=32*mm, bottomMargin=20*mm)
    story = []
    story.append(Spacer(1, 60))
    # Big title centered
    story.append(Paragraph(f'<para align="center"><font size=28 color="#FFFFFF"><b>{title}</b></font></para>', ParagraphStyle("cover_title", fontName=DEFAULT_FONT, leading=30)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f'<para align="center"><font size=12 color="#E5E7EB">{subtitle}</font></para>', ParagraphStyle("cover_sub", fontName=DEFAULT_FONT, leading=14)))
    story.append(Spacer(1, 140))
    # footer signature and confidentiality line
    footer = Paragraph(
        '<para align="center"><font size=9 color="#D1D5DB">© 2025 People Analytics Project — Confidential</font></para>',
        ParagraphStyle("footer", fontName=DEFAULT_FONT)
    )
    story.append(footer)
    # Render with a dark-blue background by drawing a rectangle via onFirstPage
    def _cover_canvas(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0B274B"))
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.restoreState()
    doc.build(story, onFirstPage=_cover_canvas)

def _build_consolidated_toc(path_out: str, consolidated_rows: List[Dict[str,str]]):
    """consolidated_rows: list of {"Module":..., "Metrics":...}"""
    doc = SimpleDocTemplate(path_out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    story = []
    story.append(Paragraph("<b>📘 Consolidated Table of Contents</b>", subheading_style))
    story.append(Spacer(1, 8))

    # Build table data
    data = [["Module", "Metrics Overview"]]
    for r in consolidated_rows:
        data.append([r.get("Module", ""), r.get("Metrics", "")])
    col_widths = [50*mm, (PAGE_WIDTH - 36*mm) - 50*mm]  # left margin accounted

    table = Table(data, colWidths=col_widths)
    # zebra style rows (skip header)
    row_bg = []
    for idx in range(len(data)):
        if idx == 0:
            row_bg.append(colors.HexColor("#E5E7EB"))
        else:
            row_bg.append(colors.whitesmoke if idx%2==1 else colors.HexColor("#F3F4F6"))
    table_style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#94A3B8")),
        ("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ])
    # add row backgrounds
    for ridx, bg in enumerate(row_bg):
        table_style.add("BACKGROUND", (0, ridx), (-1, ridx), bg)
    table.setStyle(table_style)
    story.append(table)
    doc.build(story)

def _build_section_divider(path_out: str, module: str, metrics_overview: str = ""):
    doc = SimpleDocTemplate(path_out, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=40*mm, bottomMargin=20*mm)
    story = []
    story.append(Spacer(1, 80))
    # Emoji + Title
    story.append(Paragraph(f'<para align="left"><font size=26 color="#FACC15">🧩</font> <font size=22 color="#FFFFFF"><b>{module}</b></font></para>', ParagraphStyle("sec_title", fontName=DEFAULT_FONT)))
    story.append(Spacer(1, 10))
    if metrics_overview:
        story.append(Paragraph(f'<font size=11 color="#E5E7EB">{metrics_overview}</font>', ParagraphStyle("sec_sub", fontName=DEFAULT_FONT)))
    doc.build(story, onFirstPage=lambda c,d: c.setFillColor(colors.HexColor("#071430")))

def _build_consolidated_exec_summary(path_out: str, consolidated_summary_rows: List[Dict[str,str]]):
    """Builds a neat zebra table where each row = Module / Key Insights (joined)."""
    doc = SimpleDocTemplate(path_out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    story = []
    story.append(Paragraph("<b>🧠 Consolidated Executive Summary</b>", subheading_style))
    story.append(Spacer(1, 8))

    data = [["Module", "Key Insights"]]
    for r in consolidated_summary_rows:
        key = r.get("Insights", "")
        # join long insight lines with bullet separators for readability
        if isinstance(key, list):
            key_text = " • ".join(key)
        else:
            key_text = str(key)
        data.append([r.get("Module", ""), key_text])

    col_widths = [60*mm, (PAGE_WIDTH - 36*mm) - 60*mm]
    table = Table(data, colWidths=col_widths)
    # zebra background
    table_style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#94A3B8")),
        ("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ])
    # alternate row backgrounds
    for i in range(1, len(data)):
        bg = colors.HexColor("#FFFFFF") if i%2==1 else colors.HexColor("#F9FAFB")
        table_style.add("BACKGROUND", (0,i), (-1,i), bg)
    table.setStyle(table_style)
    story.append(table)
    doc.build(story)

def _build_thankyou_page(path_out: str, signature_line: str = "Prepared with ❤️ by Amlan Mishra — People Analytics Project"):
    doc = SimpleDocTemplate(path_out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    story = []
    story.append(Spacer(1, 80))
    story.append(Paragraph(f'<para align="center"><font size=26 color="#FFFFFF"><b>Thank you</b></font></para>', ParagraphStyle("ty_title", fontName=DEFAULT_FONT)))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f'<para align="center"><font size=12 color="#E6EEF8">For reviewing the People Analytics consolidated leadership deck.</font></para>', ParagraphStyle("ty_sub", fontName=DEFAULT_FONT)))
    story.append(Spacer(1, 120))
    story.append(Paragraph(f'<para align="center"><font size=10 color="#D1D5DB">{signature_line}</font></para>', ParagraphStyle("ty_sig", fontName=DEFAULT_FONT)))
    # draw a rich background using onFirstPage
    def _ty_canvas(canvas, doc):
        canvas.saveState()
        # gradient-like rectangle (simple)
        canvas.setFillColor(colors.HexColor("#0B274B"))
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#08203A"))
        canvas.rect(18*mm, 24*mm, PAGE_WIDTH - 36*mm, PAGE_HEIGHT - 48*mm, stroke=0, fill=0)
        canvas.restoreState()
    doc.build(story, onFirstPage=_ty_canvas)

# -------------------------
# Main merge function
# -------------------------
def merge_consolidated_pdfs(output_path: str = None) -> bool:
    """
    Produces a polished consolidated PDF in this order:
      1) Cover
      2) Consolidated TOC (no page numbers)
      3) For each module:
           - Section divider (custom)
           - Module PDF content (skipping its first page)
      4) Consolidated Executive Summary (filtered)
      5) Thank you page

    Returns True if merged successfully and output_path exists.
    """
    tmp = TMP_DIR
    if output_path is None:
        output_path = os.path.join(tmp, "People_Analytics_Leadership_Deck.pdf")

    # find available module PDFs and metadata
    module_pdf_paths = {}
    module_meta = {}
    for mod in MODULES_ORDER:
        pdfp = os.path.join(tmp, f"{mod}.pdf")
        module_pdf_paths[mod] = pdfp if os.path.exists(pdfp) else None
        module_meta[mod] = _load_module_meta(tmp, mod)

    # Ensure at least one module available
    any_pdf = any(bool(p) for p in module_pdf_paths.values())
    if not any_pdf:
        raise FileNotFoundError("No module PDFs found in TMP_DIR to merge.")

    # Prepare consolidated TOC rows and consolidated exec summary rows
    toc_rows = []
    consolidated_summary_rows = []
    for mod in MODULES_ORDER:
        meta = module_meta.get(mod, {})
        metrics = meta.get("metrics_short", "")
        insights_raw = meta.get("insights", "")
        insights_list = _normalize_insights(insights_raw)
        # apply module-specific filters before consolidated exec summary
        filtered_insights = _filter_insights_for_module(mod, insights_list)
        # if nothing remains, show a placeholder
        if not filtered_insights:
            filtered_insights = ["No summary provided."]
        toc_rows.append({"Module": mod, "Metrics": metrics})
        consolidated_summary_rows.append({"Module": mod, "Insights": filtered_insights})

    # create temporary pdf pages: cover, toc, divider per module, consolidated exec summary, thank you
    tmp_files = {}
    try:
        # 1) cover
        cover_pdf = os.path.join(tmp, "consolidated_cover.pdf")
        _build_cover_pdf(cover_pdf)
        tmp_files["cover"] = cover_pdf

        # 2) consolidated TOC
        toc_pdf = os.path.join(tmp, "consolidated_toc.pdf")
        _build_consolidated_toc(toc_pdf, toc_rows)
        tmp_files["toc"] = toc_pdf

        # 3) section dividers per module (create only for modules present)
        divider_pdfs = {}
        for mod in MODULES_ORDER:
            if module_pdf_paths.get(mod):
                divp = os.path.join(tmp, f"divider_{mod}.pdf")
                metrics = module_meta.get(mod, {}).get("metrics_short", "")
                _build_section_divider(divp, mod, metrics_overview=metrics)
                divider_pdfs[mod] = divp
        tmp_files["dividers"] = divider_pdfs

        # 4) consolidated exec summary
        summary_pdf = os.path.join(tmp, "consolidated_exec_summary.pdf")
        _build_consolidated_exec_summary(summary_pdf, consolidated_summary_rows)
        tmp_files["summary"] = summary_pdf

        # 5) thank you page
        thank_pdf = os.path.join(tmp, "consolidated_thankyou.pdf")
        _build_thankyou_page(thank_pdf)
        tmp_files["thank"] = thank_pdf

        # Now merge using PdfReader / PdfWriter
        writer = PdfWriter()

        def append_pdf_to_writer(path, skip_pages=0):
            """Append all pages of a PDF to writer, optionally skipping first skip_pages."""
            if not os.path.exists(path):
                return 0
            reader = PdfReader(path)
            total = len(reader.pages)
            added = 0
            for p_idx in range(skip_pages, total):
                writer.add_page(reader.pages[p_idx])
                added += 1
            return added

        # Order: cover -> toc -> (for each module: divider + module content skipping first page) -> summary -> thankyou
        append_pdf_to_writer(tmp_files["cover"])
        append_pdf_to_writer(tmp_files["toc"])

        for mod in MODULES_ORDER:
            mod_pdf = module_pdf_paths.get(mod)
            if mod_pdf:
                # add section divider
                divp = tmp_files["dividers"].get(mod)
                if divp and os.path.exists(divp):
                    append_pdf_to_writer(divp)
                # append module pdf skipping first page (to avoid original cover redundancy)
                # if module pdf only 1 page, skipping will append 0 pages — still okay
                append_pdf_to_writer(mod_pdf, skip_pages=1)

        append_pdf_to_writer(tmp_files["summary"])
        append_pdf_to_writer(tmp_files["thank"])

        # write output
        with open(output_path, "wb") as out_f:
            writer.write(out_f)

        # Final check and return
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            raise RuntimeError("Merged PDF was created but is empty or missing.")
    finally:
        # We intentionally keep tmp module PDFs and json files untouched.
        # Remove only the transient files we created for the merge (cover/toc/dividers/summary/thank)
        for k, v in tmp_files.items():
            if isinstance(v, dict):
                for subp in v.values():
                    try:
                        os.remove(subp)
                    except Exception:
                        pass
            else:
                try:
                    os.remove(v)
                except Exception:
                    pass

# If run as script for local test
if __name__ == "__main__":
    out = os.path.join(TMP_DIR, "People_Analytics_Leadership_Deck.pdf")
    try:
        ok = merge_consolidated_pdfs(out)
        print("Merged ->", out, "OK:", ok)
    except Exception as e:
        print("Error while merging:", e)