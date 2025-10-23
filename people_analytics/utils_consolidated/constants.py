# utils_consolidated/constants.py
# Theme + layout constants for consolidated PDF builder

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

PAGE_SIZE = A4
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE

MARGINS = {
    "left": 18 * mm,
    "right": 18 * mm,
    "top": 20 * mm,
    "bottom": 20 * mm,
}

COVER = {
    "title_size": 24,
    "subtitle_size": 12,
    "title_color": colors.HexColor("#0F172A"),
    "subtitle_color": colors.HexColor("#374151"),
}

TOC = {
    "header_color": colors.HexColor("#E5E7EB"),
    "font_size": 9,
    "col_widths": [20, 110, 260, 30],
}

DEFAULT_FONT = "DejaVuSans"
FALLBACK_FONT = "Helvetica"

CHART_WIDTH_MM = 170
CHART_HEIGHT_MM = 95

# recommended DPI scale for kaleido exports (higher => larger PNG)
EXPORT = {
    "width": 1200,
    "height": 700,
    "scale": 2
}