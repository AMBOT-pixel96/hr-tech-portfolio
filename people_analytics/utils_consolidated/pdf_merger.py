# ============================================
# utils_consolidated/pdf_merger.py — v8.1 | Executive Live-Insights Build
# ============================================
import os
import io
import json
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
# 🗂️ Directory setup (consolidated queue)
# -------------------------------------------------------
TMP_DIR = "/tmp/consolidated_pdfs"
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------------------------------------
# 🧠 Font setup (graceful fallback)
# -------------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

# -------------------------------------------------------
# 📄 Utility: single-page builder (cover/divider/thankyou)
# -------------------------------------------------------
def _make_single_page_pdf(title: str, subtitle: str = "", color: str = "#1E3A8A") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    # background
    c.setFillColor(colors.HexColor(color))
    c.rect(0, 0, w, h, fill=True, stroke=False)
    # title
    c.setFillColor(colors.white)
    c.setFont(FONT_NAME, 26)
    c.drawCentredString(w / 2, h / 2 + 12 * mm, title)
    if subtitle:
        c.setFont(FONT_NAME, 12)
        c.drawCentredString(w / 2, h / 2 - 10 * mm, subtitle)
    # footer small
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(w / 2, 15, "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 📖 Consolidated TOC (zebra rows, no page numbers)
# -------------------------------------------------------
def _make_consolidated_toc(modules_ordered):
    """
    modules_ordered: list of tuples (module_name, short_metrics_desc)
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 20)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawCentredString(w / 2, h - 60, "📖 Consolidated Table of Contents")

    y = h - 100
    stripe_colors = [colors.white, colors.HexColor("#F3F4F6")]
    for i, (name, desc) in enumerate(modules_ordered):
        bg = stripe_colors[i % 2]
        c.setFillColor(bg)
        c.rect(40, y - 22, w - 80, 28, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont(FONT_NAME, 12)
        c.drawString(52, y - 8, name)
        c.setFont(FONT_NAME, 10)
        c.drawString(180, y - 8, desc)
        y -= 36
        if y < 80:
            c.showPage()
            y = h - 80

    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawCentredString(w / 2, 25, "Generated " + datetime.now().strftime("%b'%y"))
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 🧾 Consolidated Executive Summary (reads module metadata JSONs)
# -------------------------------------------------------
def _make_consolidated_summary(modules_list):
    """
    modules_list: list of module base names, e.g. ["Workforce","Performance",...]
    Each module can store a metadata JSON at TMP_DIR/<Module>.json with:
      {"insights": "concise insight line", "metrics_short": "short metrics description (optional)"}
    """
    # Collect rows
    rows = []
    for m in modules_list:
        meta_path = os.path.join(TMP_DIR, f"{m}.json")
        insight = "No summary provided."
        metrics_short = ""
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    insight = data.get("insights", insight)
                    metrics_short = data.get("metrics_short", "")
        except Exception:
            insight = "Could not read module metadata."
        rows.append((m, metrics_short or insight))

    # Render zebra table-like pages
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont(FONT_NAME, 20)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawCentredString(w / 2, h - 60, "📊 Consolidated Executive Summary")

    y = h - 100
    stripe_colors = [colors.white, colors.HexColor("#F3F4F6")]
    for i, (module, text) in enumerate(rows):
        bg = stripe_colors[i % 2]
        c.setFillColor(bg)
        c.rect(40, y - 22, w - 80, 28, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont(FONT_NAME, 12)
        c.drawString(52, y - 8, module)
        c.setFont(FONT_NAME, 10)
        # Truncate if very long, keep it neat
        display_text = (text[:150] + "...") if len(text) > 150 else text
        c.drawString(180, y - 8, display_text)
        y -= 36
        if y < 80:
            c.showPage()
            y = h - 80

    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawCentredString(w / 2, 25, "Amalgamated insights • " + datetime.now().strftime("%b'%y"))
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 🧩 Main merge function (public)
# -------------------------------------------------------
def merge_consolidated_pdfs(output_filename="People_Analytics_Leadership_Deck.pdf"):
    """
    Merges PDFs in TMP_DIR into a single consolidated deck.
    - Uses <Module>.json metadata files for the consolidated summary if present.
    - Skips the first page of each module's PDF (assumes it is a module cover).
    - Inserts cover -> consolidated TOC -> (divider + module pages) -> consolidated summary -> thank you page.
    """
    st.markdown("### 🧩 Consolidation Summary")
    pdf_files = [f for f in os.listdir(TMP_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.info("⚙️ No PDFs found in consolidated queue.")
        return False

    pdf_files.sort()  # deterministic order

    # Define ordered modules & short descriptions (used for consolidated TOC)
    modules_ordered = [
        ("Workforce", "Total Employees, Female %, Job Levels"),
        ("Performance", "Avg Rating, Std Dev, Top Performers %"),
        ("Engagement", "Avg Index, High Engagement %, Responses"),
        ("Compensation", "Avg CTC, Avg Bonus %"),
        ("Attrition", "Attrition %, Avg Tenure")
    ]
    module_names = [m for m, _ in modules_ordered]

    out_path = os.path.join(TMP_DIR, output_filename)
    writer = PdfWriter()

    # 1) Cover page
    try:
        cover = PdfReader(io.BytesIO(_make_single_page_pdf("People Analytics Leadership Deck", datetime.now().strftime("%b'%y"), "#0F172A")))
        writer.append(cover)
    except Exception as e:
        st.error(f"⚠️ Failed to generate cover page: {e}")
        return False

    # 2) Consolidated TOC
    try:
        toc_pdf = PdfReader(io.BytesIO(_make_consolidated_toc(modules_ordered)))
        writer.append(toc_pdf)
    except Exception as e:
        st.error(f"⚠️ Failed to generate consolidated TOC: {e}")
        return False

    # 3) Each module section (divider + module pages; skip first page of module pdf)
    summary_rows_for_generation = []  # used later to build consolidated summary using JSONs
    for fname in pdf_files:
        base = os.path.splitext(fname)[0]  # e.g., "Attrition_Analytics_Executive_Report" or "Attrition"
        # Try to map to module short name by matching known module names
        matched_module = None
        for mod in module_names:
            if mod.lower() in base.lower():
                matched_module = mod
                break
        if matched_module is None:
            # fallback: use cleaned basename first token
            matched_module = base.split("_")[0].title()

        st.write(f"📄 Adding section: {matched_module}")
        # Divider page
        try:
            divider = PdfReader(io.BytesIO(_make_single_page_pdf(matched_module, "Module Summary", "#1E3A8A")))
            writer.append(divider)
        except Exception as e:
            st.warning(f"⚠️ Could not create divider for {matched_module}: {e}")

        # Append module PDF pages (skipping its first page, which is assumed a module cover)
        try:
            reader = PdfReader(os.path.join(TMP_DIR, fname))
            for i, page in enumerate(reader.pages):
                if i == 0:
                    continue  # skip redundant module cover
                writer.add_page(page)
        except Exception as e:
            st.error(f"⚠️ Could not append pages for {fname}: {e}")

        # collect for summary creation (metadata-driven)
        summary_rows_for_generation.append(matched_module)

    # 4) Consolidated Executive Summary (reads module jsons)
    try:
        st.write("🧠 Adding Consolidated Executive Summary...")
        summary_pdf = PdfReader(io.BytesIO(_make_consolidated_summary(summary_rows_for_generation)))
        writer.append(summary_pdf)
    except Exception as e:
        st.error(f"⚠️ Failed to add consolidated summary: {e}")

    # 5) Thank You page
    try:
        thanks_pdf = PdfReader(io.BytesIO(_make_single_page_pdf("Thank You 💼", "Prepared by People Analytics — 2025", "#0F172A")))
        writer.append(thanks_pdf)
    except Exception as e:
        st.warning(f"⚠️ Could not add Thank You page: {e}")

    # 6) Write output
    try:
        with open(out_path, "wb") as out_f:
            writer.write(out_f)
        st.success("✅ Consolidated Leadership Deck created successfully!")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Final Consolidated Deck", f, file_name=output_filename, mime="application/pdf")
        return True
    except Exception as e:
        st.error(f"❌ Failed to write merged PDF: {e}")
        return False

# -------------------------------------------------------
# 🧹 Convenience: clear consolidated queue
# -------------------------------------------------------
def clear_consolidated_queue():
    try:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR, exist_ok=True)
        st.success("✅ Consolidated queue cleared.")
    except Exception as e:
        st.error(f"⚠️ Failed to clear queue: {e}")