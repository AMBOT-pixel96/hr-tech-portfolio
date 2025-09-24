from pathlib import Path
import subprocess
import sys

# NOTE: update this filename if your main notebook name is different
notebook = Path("notebooks/Attrition_ModelComparison.ipynb")
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

# nbconvert will put a file named Attrition_Project_Summary.pdf in reports/
cmd = [
    "jupyter", "nbconvert",
    "--to", "pdf",
    "--TemplateExporter.exclude_input=True",
    str(notebook),
    "--output-dir", str(output_dir),
    "--output", "Attrition_Project_Summary"
]

print("📄 Running nbconvert...")
subprocess.run(cmd, check=True)
print(f"✅ PDF should be at: {output_dir / 'Attrition_Project_Summary.pdf'}")