import os
from pathlib import Path
from nbconvert import HTMLExporter
import nbformat
from weasyprint import HTML

# Look for notebooks in repo root
NOTEBOOKS_DIR = Path(".")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

def convert_notebook_to_html(notebook_path, html_path):
    """Convert a Jupyter notebook to HTML."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    (body, _) = html_exporter.from_notebook_node(nb)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)

def export_html_to_pdf(html_path, pdf_path):
    """Convert HTML to PDF using WeasyPrint (Python API)."""
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))

def main():
    # Find notebooks in repo root
    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("❌ No notebooks found in repo root")

    latest_notebook = max(notebooks, key=os.path.getmtime)
    print(f"📒 Latest project notebook found: {latest_notebook.name}")

    html_file = REPORTS_DIR / "temp.html"
    stable_pdf = REPORTS_DIR / "Attrition_Project_Summary.pdf"
    versioned_pdf = REPORTS_DIR / f"{latest_notebook.stem}_{os.urandom(4).hex()}.pdf"

    # Step 1: Notebook → HTML
    convert_notebook_to_html(latest_notebook, html_file)

    # Step 2: HTML → PDF
    export_html_to_pdf(html_file, stable_pdf)
    export_html_to_pdf(html_file, versioned_pdf)

    print(f"✅ PDF reports created: {stable_pdf} and {versioned_pdf}")

if __name__ == "__main__":
    main()