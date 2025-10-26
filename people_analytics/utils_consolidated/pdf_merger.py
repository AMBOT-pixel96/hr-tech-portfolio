# ============================================
# utils_consolidated/pdf_merger.py — v8.0 | Executive Perfection Build
# ============================================
import os
import io
import shutil
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------------------------------------------------------
# 🗂️ Directory setup
# -------------------------------------------------------
TMP_DIR = "/tmp/consolidated_pdfs"
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------------------------------------
# 🧠 Font setup
# -------------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

# -------------------------------------------------------
# 📄 Utility: single-page builder (cover/divider)
# -------------------------------------------------------
def _make_single_page_pdf(title: str, subtitle: str = "", color: str = "#1E3A8A") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFillColor(colors.HexColor(color))
    c.rect(0, 0, w, h, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont(FONT_NAME, 24)
    c.drawCentredString(w / 2, h / 2 + 10 * mm, title)
    if subtitle:
        c.setFont(FONT_NAME, 14)
        c.drawCentredString(w / 2, h / 2 - 10 * mm, subtitle)
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(w / 2, 15,
        "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 📖 Consolidated TOC (Module → Metrics)
# -------------------------------------------------------
def _make_consolidated_toc():
    toc_data = [
        ("Workforce",    "Total Employees, Female %, Job Levels"),
        ("Performance",  "Avg Rating, Std Dev, Top Performers %"),
        ("Engagement",   "Avg Index, High Engagement %, Responses"),
        ("Compensation", "Avg CTC, Avg Bonus %"),
        ("Attrition",    "Attrition %, Avg Tenure"),
    ]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 22)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawCentredString(w / 2, h - 60, "📖 Consolidated Table of Contents")

    y = h - 110
    stripes = [colors.white, colors.HexColor("#ECECEC")]
    for i, (m, desc) in enumerate(toc_data):
        bg = stripes[i % 2]
        c.setFillColor(bg)
        c.rect(40, y - 20, w - 80, 25, fill=True, stroke=False)
        c.setFillColor(colors.black)
        c.setFont(FONT_NAME, 12)
        c.drawString(60, y - 10, m)
        c.setFont(FONT_NAME, 10)
        c.drawString(180, y - 10, desc)
        y -= 28

    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawCentredString(w / 2, 25, "Generated on " + datetime.now().strftime("%b'%y"))
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 🧾 Consolidated Executive Summary Table
# -------------------------------------------------------
def _make_consolidated_summary(summary_rows):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 22)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawCentredString(w / 2, h - 60, "📊 Consolidated Executive Summary")

    y = h - 110
    stripes = [colors.white, colors.HexColor("#F3F4F6")]
    for i, (module, insight) in enumerate(summary_rows):
        bg = stripes[i % 2]
        c.setFillColor(bg)
        c.rect(40, y - 20, w - 80, 25, fill=True, stroke=False)
        c.setFillColor(colors.black)
        c.setFont(FONT_NAME, 12)
        c.drawString(60, y - 10, module)
        c.setFont(FONT_NAME, 10)
        c.drawString(180, y - 10, insight)
        y -= 28
        if y < 80:  # new page
            c.showPage()
            y = h - 80

    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawCentredString(w / 2, 25,
        "Amalgamated module insights • Generated " + datetime.now().strftime("%b'%y"))
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 🧩 Merge PDFs into one executive deck
# -------------------------------------------------------
def merge_consolidated_pdfs(output_filename="People_Analytics_Leadership_Deck.pdf"):
    st.markdown("### 🧩 Consolidation Summary")
    pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.info("⚙️ No PDFs found in consolidated queue.")
        return False
    pdf_files.sort()
    out_path = os.path.join(TMP_DIR, output_filename)
    writer = PdfWriter()

    # 1️⃣ Cover Page
    cover = PdfReader(io.BytesIO(_make_single_page_pdf(
        "People Analytics Leadership Deck",
        datetime.now().strftime("%b'%y"),
        "#0F172A"
    )))
    writer.append(cover)

    # 2️⃣ Consolidated TOC
    toc = PdfReader(io.BytesIO(_make_consolidated_toc()))
    writer.append(toc)

    # 3️⃣ Module Sections (+ skip first page)
    summary_rows = []
    for f in pdf_files:
        name = os.path.splitext(f)[0].replace("_", " ")
        st.write(f"📄 Adding section: {name}")
        divider = PdfReader(io.BytesIO(_make_single_page_pdf(
            name.title(), "Module Summary", "#1E3A8A"
        )))
        writer.append(divider)

        try:
            reader = PdfReader(os.path.join(TMP_DIR, f))
            for i, page in enumerate(reader.pages):
                if i == 0:  # skip module cover
                    continue
                writer.add_page(page)
        except Exception as e:
            st.error(f"⚠️ Could not merge {name}: {e}")

        # synthetic high-level summary line (mocked; would come from data)
        base = name.split()[0]
        summaries = {
            "Workforce": "Total Employees • Female % • Job Levels",
            "Performance": "Avg Rating • Top Performers % • Avg CTC",
            "Engagement": "Avg Index • Highly Engaged % • Responses",
            "Compensation": "Avg CTC • Avg Bonus %",
            "Attrition": "Attrition % • Avg Tenure Months"
        }
        summary_rows.append((base, summaries.get(base, "—")))

    # 4️⃣ Consolidated Executive Summary
    st.write("🧠 Adding Consolidated Executive Summary page…")
    summary_pdf = PdfReader(io.BytesIO(_make_consolidated_summary(summary_rows)))
    writer.append(summary_pdf)

    # 5️⃣ Thank You Page
    st.write("🙏 Adding Thank You page…")
    thanks = PdfReader(io.BytesIO(_make_single_page_pdf(
        "Thank You 💼",
        "Prepared with ❤️ by Amlan Mishra — People Analytics Project (2025)",
        "#0F172A"
    )))
    writer.append(thanks)

    # 6️⃣ Save & Download
    with open(out_path, "wb") as f:
        writer.write(f)
    st.success("✅ Consolidated Leadership Deck created successfully!")
    with open(out_path, "rb") as f:
        st.download_button(
            "⬇️ Download Final Consolidated Deck",
            f,
            file_name=output_filename,
            mime="application/pdf"
        )
    return True