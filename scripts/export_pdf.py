from pathlib import Path
import subprocess
import datetime

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

# Find the most recently modified notebook
notebooks = sorted(
    Path(".").glob("*.ipynb"),   # look in repo root instead of /notebooks
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not notebooks:
    raise FileNotFoundError("❌ No notebooks found in /notebooks folder")

latest = notebooks[0]
print(f"📓 Latest notebook found: {latest}")

# File names
stable_pdf = output_dir / "Attrition_Project_Summary.pdf"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
archived_pdf = output_dir / f"{latest.stem}_{timestamp}.pdf"
temp_html = output_dir / "temp.html"

def export_html_to_pdf(input_nb, output_pdf):
    # Step 1: Notebook → HTML
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "html",
        "--TemplateExporter.exclude_input=True",
        str(input_nb),
        "--output", str(temp_html)
    ], check=True)

    # Step 2: HTML → PDF
    subprocess.run([
        "weasyprint", str(temp_html), str(output_pdf)
    ], check=True)

    print(f"✅ PDF created at {output_pdf}")

# Export both stable and archived
export_html_to_pdf(latest, stable_pdf)
export_html_to_pdf(latest, archived_pdf)