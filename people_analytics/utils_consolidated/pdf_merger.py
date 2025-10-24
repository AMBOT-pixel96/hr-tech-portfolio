# ============================================
# utils_consolidated/pdf_merger.py — v2.1 | Executive Boardroom Edition (Stable)
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
def merge_pdfs(output_path: str = os.path.join(TMP_DIR, "People_Analytics_Leadership_Deck.pdf")) -> bool:
    """
    Merges all PDFs from TMP_DIR into one executive deck with dividers.
    Returns True on success, False otherwise.
    """
    pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("⚙️ No PDFs found in consolidated queue. Add modules first.")
        return False

    pdf_files.sort()  # Consistent alphabetical order

    writer = PdfWriter()

    # -------------------------------------------------------
    # 🧠 1. Cover Page
    # -------------------------------------------------------
    try:
        cover_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
            "People Analytics Leadership Deck",
            f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            "#0F172A"
        )))
        writer.append(cover_pdf)
    except Exception as e:
        st.error(f"⚠️ Failed to add cover page: {e}")
        return False

    # -------------------------------------------------------
    # 🧩 2. Append each module report with divider
    # -------------------------------------------------------
    for f in pdf_files:
        section_name = os.path.splitext(f)[0].replace("_", " ")
        try:
            # Divider page
            divider_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
                section_name.title(),
                "Module Summary",
                "#1E3A8A"
            )))
            writer.append(divider_pdf)

            # Module PDF
            reader = PdfReader(os.path.join(TMP_DIR, f))
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            st.error(f"⚠️ Could not merge {f}: {e}")

    # -------------------------------------------------------
    # 🧾 3. Write final merged PDF
    # -------------------------------------------------------
    try:
        with open(output_path, "wb") as out:
            writer.write(out)
        st.success("✅ Consolidated Leadership Deck created successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to write merged PDF: {e}")
        return False

# -------------------------------------------------------
# 🧹 Optional Cleanup Utility
# -------------------------------------------------------
def clear_consolidated_queue():
    """Safely clears the consolidated temp folder."""
    try:
        shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR, exist_ok=True)
        st.success("✅ Consolidated queue cleared successfully.")
    except Exception as e:
        st.error(f"⚠️ Failed to clear queue: {e}")