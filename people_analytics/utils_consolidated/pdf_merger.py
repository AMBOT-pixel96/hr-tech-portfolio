# ============================================
# utils_consolidated/pdf_merger.py — v4.5 | Executive Boardroom + Thank You Finale
# ============================================
import os
import io
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------------------------------------------------------
# 🗂️ Directory setup
# -------------------------------------------------------
TMP_DIR = os.path.join("/tmp", "consolidated_pdfs")
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
# 📄 Helper: generate single-page PDF (cover, divider, or thank-you)
# -------------------------------------------------------
def _make_single_page_pdf(title: str, subtitle: str = "", color: str = "#1E3A8A", text_color: str = "white") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor(color))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    c.setFillColor(colors.HexColor(text_color))
    c.setFont(FONT_NAME, 24)
    c.drawCentredString(width / 2, height / 2 + 10 * 10, title)
    if subtitle:
        c.setFont(FONT_NAME, 14)
        c.drawCentredString(width / 2, height / 2 - 10 * 10, subtitle)
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(width / 2, 15, "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 📦 Merge PDFs into one executive deck with a Thank You page
# -------------------------------------------------------
def merge_pdfs():
    """Merge all PDFs in TMP_DIR into a single consolidated deck, with divider pages and a 'Thank You' finale."""
    try:
        pdf_files = [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            st.warning("⚠️ No module PDFs found. Please generate module reports first.")
            return

        pdf_files.sort()  # Consistent order
        writer = PdfWriter()

        # -------------------------------------------------------
        # 🧠 Cover Page
        # -------------------------------------------------------
        cover_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
            "People Analytics Leadership Deck",
            f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            "#0F172A"
        )))
        writer.append(cover_pdf)

        # -------------------------------------------------------
        # 🧩 Merge each module with divider
        # -------------------------------------------------------
        for path in pdf_files:
            section_name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
            try:
                divider_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
                    section_name.title(),
                    "Module Summary",
                    "#1E3A8A"
                )))
                writer.append(divider_pdf)

                reader = PdfReader(path)
                for page in reader.pages:
                    writer.add_page(page)
            except Exception as e:
                st.error(f"⚠️ Could not merge {path}: {e}")

        # -------------------------------------------------------
        # 🎬 Thank You Finale Page
        # -------------------------------------------------------
        thank_you_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
            "Thank You 💼",
            "Prepared with ❤️ by Amlan Mishra | People Analytics Project — 2025",
            "#312E81",  # Deep indigo-violet
            "white"
        )))
        writer.append(thank_you_pdf)

        # -------------------------------------------------------
        # 💾 Write to memory and enable download
        # -------------------------------------------------------
        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        st.success("✅ Consolidated Leadership Deck created successfully!")
        st.download_button(
            "⬇️ Download Consolidated HR Leadership Deck (PDF)",
            buffer,
            file_name="People_Analytics_Leadership_Deck.pdf",
            mime="application/pdf",
        )

        return True

    except Exception as e:
        st.error(f"❌ Failed to merge PDFs: {e}")
        return False