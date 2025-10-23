# utils_consolidated/toc_helper.py
"""
Simple TOC helper: builds a table-friendly TOC array for ReportLab.
"""

from .constants import TOC

def build_toc_entries(data_blocks):
    """
    data_blocks: list of dicts with keys 'title' and 'desc'
    returns: list of rows including header suitable for reportlab Table
    """
    header = ["#", "Section", "Description", "Page"]
    rows = [header]
    for i, b in enumerate(data_blocks, start=1):
        rows.append([i, b.get("title", ""), b.get("desc", ""), str(i + 2)])  # approximate pages
    return rows

def toc_colwidths():
    return TOC.get("col_widths", [20, 100, 250, 30])