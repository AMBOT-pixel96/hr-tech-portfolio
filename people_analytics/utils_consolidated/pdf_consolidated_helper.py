# ============================================
# utils_consolidated/pdf_consolidated_helper.py
# v6.0 — WeasyPrint Streamlit Edition (No Kaleido, No /tmp)
# ============================================

import io
import streamlit as st
from datetime import datetime
from weasyprint import HTML

# ---------------------------------------------------
# 🧾 Pure HTML → PDF builder
# ---------------------------------------------------
def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
    if not modules_payload:
        st.warning("⚠️ No module data available for PDF generation.")
        return

    if st.button("🧾 Generate Consolidated Executive Deck", use_container_width=True):
        try:
            html_parts = [
                f"""
                <html><head><meta charset='utf-8'>
                <style>
                    body {{ font-family: 'Open Sans', sans-serif; color:#111; }}
                    h1,h2,h3 {{ color:#1E3A8A; }}
                    table {{ border-collapse: collapse; width:100%; margin:10px 0; }}
                    th, td {{ border:1px solid #ccc; padding:4px 6px; font-size:10px; }}
                    th {{ background:#E5E7EB; }}
                    .cover {{ text-align:center; margin-top:100px; }}
                </style></head><body>
                <div class='cover'>
                    <h1>{report_title}</h1>
                    <p><b>People Analytics Leadership Deck</b></p>
                    <p>Generated {datetime.now().strftime('%d %b %Y, %H:%M %p')}</p>
                    <hr>
                </div>
                """
            ]

            # TOC
            html_parts.append("<h2>Table of Contents</h2><ol>")
            for i, mod in enumerate(modules_payload, 1):
                html_parts.append(f"<li><b>{mod.get('module_name','')}</b> — {mod.get('module_desc','')}</li>")
            html_parts.append("</ol><hr>")

            # Modules
            for mod in modules_payload:
                html_parts.append(f"<h2>{mod.get('module_name')}</h2>")
                html_parts.append(f"<p><i>{mod.get('module_desc')}</i></p>")
                for block in mod.get("data_blocks", []):
                    html_parts.append(f"<h3>{block.get('title','')}</h3>")
                    html_parts.append(f"<p>{block.get('desc','')}</p>")
                    df = block.get("df")
                    if df is not None and not df.empty:
                        html_parts.append(df.to_html(index=False, border=0))
                    insights = block.get("insights", [])
                    if insights:
                        joined = " • ".join(str(i) for i in insights)
                        html_parts.append(f"<p><b>Insights:</b> {joined}</p>")
                    html_parts.append("<hr>")
            
            html_parts.append(
                "<footer><p style='text-align:center;font-size:9px;color:#666;'>"
                "Prepared with ❤️ by People Analytics Project — 2025</p></footer></body></html>"
            )

            final_html = "".join(html_parts)
            pdf_bytes = HTML(string=final_html).write_pdf()

            st.success("✅ PDF generated successfully (WeasyPrint engine).")
            st.download_button(
                "⬇️ Download Consolidated Deck (PDF)",
                pdf_bytes,
                file_name=f"{filename_prefix}_Leadership_Deck.pdf",
                mime="application/pdf",
            )

        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")