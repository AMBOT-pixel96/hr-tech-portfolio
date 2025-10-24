# ============================================
# utils_consolidated/pdf_merger.py — v2.0 | Executive Boardroom Edition
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
# 🧠 Fonts
# -------------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

# -------------------------------------------------------
# 📄 Helper: generate single-page PDF (cover or divider)
# -------------------------------------------------------
def _make_single_page_pdf(title: str, subtitle: str = "", color: str = "#1E3A8A") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor(color))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont(FONT_NAME, 24)
    c.drawCentredString(width / 2, height / 2 + 10 * mm, title)
    if subtitle:
        c.setFont(FONT_NAME, 14)
        c.drawCentredString(width / 2, height / 2 - 10 * mm, subtitle)
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(width / 2, 15, "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 📦 Merge PDFs into one executive deck
# -------------------------------------------------------
def merge_consolidated_pdfs(output_filename: str = "People_Analytics_Leadership_Deck.pdf"):
    """
    Merges all PDFs from TMP_DIR into one, inserting section divider pages.
    """
    st.markdown("### 🧩 Consolidation Summary")

    pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.info("⚙️ No PDFs found in consolidated queue. Add modules first.")
        return

    pdf_files.sort()  # keep consistent ordering

    st.write(f"Found {len(pdf_files)} reports:")
    for f in pdf_files:
        st.write(f"• {f}")

    output_path = os.path.join(TMP_DIR, output_filename)
    writer = PdfWriter()

    # -------------------------------------------------------
    # 🧠 1. Cover Page
    # -------------------------------------------------------
    st.write("📘 Adding cover page...")
    cover_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
        "People Analytics Leadership Deck",
        f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        "#0F172A"
    )))
    writer.append(cover_pdf)

    # -------------------------------------------------------
    # 🧩 2. Add all reports with dividers
    # -------------------------------------------------------
    for f in pdf_files:
        section_name = os.path.splitext(f)[0].replace("_", " ")
        st.write(f"📄 Adding section: {section_name}")

        # Divider
        divider_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
            section_name.title(),
            "Module Summary",
            "#1E3A8A"
        )))
        writer.append(divider_pdf)

        # Actual report
        try:
            reader = PdfReader(os.path.join(TMP_DIR, f))
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            st.error(f"⚠️ Could not merge {f}: {e}")

    # -------------------------------------------------------
    # 🧾 3. Write final file
    # -------------------------------------------------------
    with open(output_path, "wb") as out:
        writer.write(out)

    st.success(f"✅ Consolidated Leadership Deck created: {output_filename}")
    with open(output_path, "rb") as f:
        st.download_button(
            "⬇️ Download Final Consolidated Deck",
            f,
            file_name=output_filename,
            mime="application/pdf"
        )

    # -------------------------------------------------------
    # 🧹 Optional cleanup prompt
    # -------------------------------------------------------
    if st.button("🧹 Clear Consolidated Queue", use_container_width=True):
        try:
            shutil.rmtree(TMP_DIR)
            os.makedirs(TMP_DIR, exist_ok=True)
            st.success("✅ Consolidated queue cleared successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to clear queue: {e}")