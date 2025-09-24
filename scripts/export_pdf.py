import os
import subprocess
import hashlib
from pathlib import Path
from nbconvert import HTMLExporter
import nbformat
from weasyprint import HTML
import datetime

# Look for notebooks in repo root
NOTEBOOKS_DIR = Path(".")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def git_commit_time(path):
    """Return unix epoch (int) of last git commit touching the file, or None."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return int(out)
    except Exception:
        return None

def format_ts(ts):
    if ts is None:
        return "None"
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

def convert_notebook_to_html(notebook_path, html_path):
    """Convert a Jupyter notebook to HTML (code cells excluded)."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True  # 🚀 hide code cells
    (body, _) = html_exporter.from_notebook_node(nb)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)

def export_html_to_pdf(html_path, pdf_path):
    """Convert HTML to PDF using WeasyPrint (Python API)."""
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))

def main():
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("❌ No notebooks found in repo root")

    # Gather info for debug
    info = []
    for nb in notebooks:
        mtime = int(nb.stat().st_mtime)
        git_ts = git_commit_time(nb)
        info.append((nb, git_ts, mtime))

    # Print debug table
    print("🧾 Notebooks found (name | git_commit_ts | git_commit_readable | mtime_readable):")
    for nb, git_ts, mtime in info:
        print(f" - {nb.name} | {git_ts or 'None'} | {format_ts(git_ts)} | {datetime.datetime.utcfromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Choose by git commit time if available, else by mtime
    git_available = [t for t in info if t[1] is not None]
    if git_available:
        latest = max(git_available, key=lambda t: t[1])[0]
        reason = "git commit time"
    else:
        latest = max(info, key=lambda t: t[2])[0]
        reason = "file mtime (git info not available)"

    print(f"🔎 Selected notebook: {latest.name} (based on {reason})")

    # Paths
    html_file = REPORTS_DIR / "temp.html"
    stable_pdf = REPORTS_DIR / "Attrition_Project_Summary.pdf"
    nb_hash = file_hash(latest)
    archived_pdf = REPORTS_DIR / f"{latest.stem}_{nb_hash}.pdf"

    # Convert → HTML → PDF
    print(f"📄 Converting {latest} -> HTML")
    convert_notebook_to_html(latest, html_file)

    print(f"📄 Rendering HTML -> PDF (stable): {stable_pdf}")
    export_html_to_pdf(html_file, stable_pdf)

    if not archived_pdf.exists():
        print(f"📦 Creating archive: {archived_pdf}")
        export_html_to_pdf(html_file, archived_pdf)
    else:
        print(f"ℹ️ Archive exists for this notebook hash, skipping archive creation: {archived_pdf}")

    print(f"✅ Done. Stable: {stable_pdf} ; Archive (if created): {archived_pdf}")

if __name__ == "__main__":
    main()        nb = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, _) = html_exporter.from_notebook_node(nb)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)

def export_html_to_pdf(html_path, pdf_path):
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))

def main():
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("❌ No notebooks found in repo root")

    # Gather info for debug
    info = []
    for nb in notebooks:
        mtime = int(nb.stat().st_mtime)
        git_ts = git_commit_time(nb)
        info.append((nb, git_ts, mtime))

    # Print debug table
    print("🧾 Notebooks found (name | git_commit_ts | git_commit_readable | mtime_readable):")
    for nb, git_ts, mtime in info:
        print(f" - {nb.name} | {git_ts or 'None'} | {format_ts(git_ts)} | {datetime.datetime.utcfromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Choose by git commit time if available, else by mtime
    # (Prefer latest git commit timestamp among files that have git info)
    git_available = [t for t in info if t[1] is not None]
    if git_available:
        latest = max(git_available, key=lambda t: t[1])[0]
        reason = "git commit time"
    else:
        latest = max(info, key=lambda t: t[2])[0]
        reason = "file mtime (git info not available)"

    print(f"🔎 Selected notebook: {latest.name} (based on {reason})")

    # Prepare paths
    html_file = REPORTS_DIR / "temp.html"
    stable_pdf = REPORTS_DIR / "Attrition_Project_Summary.pdf"

    # Archive name based on file hash (content-based)
    nb_hash = file_hash(latest)
    archived_pdf = REPORTS_DIR / f"{latest.stem}_{nb_hash}.pdf"

    # Convert -> HTML -> PDF
    print(f"📄 Converting {latest} -> HTML")
    convert_notebook_to_html(latest, html_file)

    print(f"📄 Rendering HTML -> PDF (stable): {stable_pdf}")
    export_html_to_pdf(html_file, stable_pdf)

    if not archived_pdf.exists():
        print(f"📦 Creating archive: {archived_pdf}")
        export_html_to_pdf(html_file, archived_pdf)
    else:
        print(f"ℹ️ Archive exists for this notebook hash, skipping archive creation: {archived_pdf}")

    print(f"✅ Done. Stable: {stable_pdf} ; Archive (if created): {archived_pdf}")

if __name__ == "__main__":
    main()
