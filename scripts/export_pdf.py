from pathlib import Path
import subprocess
import datetime
import hashlib

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

# Look for notebooks in repo root
notebooks = sorted(
    Path(".").glob("*.ipynb"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not notebooks:
    raise FileNotFoundError("❌ No notebooks found in repo root")

latest = notebooks[0]
print(f"📓 Latest project notebook found: {latest}")

# Paths
stable_pdf = output_dir / "Attrition_Project_Summary.pdf"
temp_html = output_dir / "temp.html"

# Create a hash of the notebook (content-based versioning)
def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

hash_value = file_hash(latest)
archived_pdf = output_dir / f"{latest.stem}_{hash_value}.pdf"

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

    print(f"✅ PDF created: {output_pdf}")

# Always update stable
export_html_to_pdf(latest, stable_pdf)

# Only create archive if this hash file doesn't already exist
if not archived_pdf.exists():
    export_html_to_pdf(latest, archived_pdf)
    print("📦 Archived new version")
else:
    print("ℹ️ No changes detected — archive not updated")    # Step 2: HTML → PDF
    subprocess.run([
        "weasyprint", str(temp_html), str(output_pdf)
    ], check=True)

    print(f"✅ PDF created at {output_pdf}")

# Export both stable and archived
export_html_to_pdf(latest, stable_pdf)
export_html_to_pdf(latest, archived_pdf)
