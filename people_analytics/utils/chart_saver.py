# ============================================
# utils/chart_saver.py — v3.1.0 | Kaleido-forced + Safe-Restore
# ============================================
import os
import time
import logging
from typing import Optional, Callable

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# configure simple logger for debug (will show in Streamlit logs)
logger = logging.getLogger("chart_saver")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _ensure_tmp_dir(tmp_dir: str = "temp_charts") -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def _safe_filename(title: str) -> str:
    # simple filename sanitization
    name = "".join(c if (c.isalnum() or c in (" ", "-", "_")) else "_" for c in title).strip()
    return name.replace(" ", "_")[:180]


def save_chart_image(
    title: str,
    fig,
    tmp_dir: str = "temp_charts",
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
    wait_secs: float = 0.25,
    engine: str = "kaleido"
) -> Optional[str]:
    """
    Save a Plotly figure to PNG with robust theme handling and synchronous file checks.

    Key behaviours:
      • Temporarily enforce a white background + black text for the saved image
        so colors and axis labels are preserved in exported PDFs.
      • Restore only the small set of properties we change (so we don't lose user
        custom layout other code may set).
      • Force Kaleido as the engine and verify file existence before returning.
      • Returns absolute path to saved PNG or None if save failed.

    Usage:
        path = save_chart_image("Avg CTC by Level", fig)
    """
    try:
        tmp_dir = _ensure_tmp_dir(tmp_dir)
        fname = f"{_safe_filename(title)}.png"
        img_path = os.path.join(tmp_dir, fname)

        # Save snapshot of only the properties we will modify so we can restore them.
        prev_template = getattr(fig.layout, "template", None)
        prev_paper_bg = getattr(fig.layout, "paper_bgcolor", None)
        prev_plot_bg = getattr(fig.layout, "plot_bgcolor", None)
        prev_font = getattr(fig.layout, "font", None)
        prev_xaxis = fig.layout.to_plotly_json().get("xaxis", None)  # keep axis definitions safe
        prev_yaxis = fig.layout.to_plotly_json().get("yaxis", None)

        # --- Apply minimal PDF-friendly layout overrides ---
        # Use plotly_white template + explicit white paper/plot background + black font
        try:
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#000000"),
            )
        except Exception as e:
            # not fatal — continue but warn
            logger.warning("Could not set layout overrides: %s", e)

        # minor trace touches to ensure outline visibility (helpful for stacked bars)
        try:
            for tr in fig.data:
                # handle markers (bars, scatter) and lines to ensure contrasts remain visible
                if hasattr(tr, "marker") and tr.marker is not None:
                    # do not overwrite color — only ensure a thin line for definition
                    if getattr(tr.marker, "line", None) is None:
                        tr.marker.line = dict(width=0.6, color="#DDDDDD")
                if hasattr(tr, "line") and tr.line is not None:
                    # ensure lines have at least some width for visibility
                    if getattr(tr.line, "width", None) in (None, 0):
                        tr.line.width = getattr(tr.line, "width", 1)
        except Exception as e:
            logger.debug("Trace adjustments skipped: %s", e)

        # --- Write image using kaleido engine explicitly ---
        # Use engine param to make sure we don't fall back to or use other renderers.
        try:
            fig.write_image(img_path, width=width, height=height, scale=scale, engine=engine)
        except TypeError:
            # older versions of plotly may not accept engine param — fallback
            fig.write_image(img_path, width=width, height=height, scale=scale)
        except Exception as e:
            logger.exception("Failed to write image using kaleido: %s", e)
            st.warning(f"⚠️ Could not export chart image for '{title}': {e}")
            # Attempt to restore layout before returning
            try:
                _restore_layout(fig, prev_template, prev_paper_bg, prev_plot_bg, prev_font, prev_xaxis, prev_yaxis)
            except Exception:
                pass
            return None

        # small synchronous buffer to ensure disk is flushed
        time.sleep(wait_secs)

        # verify
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            logger.error("Saved image missing or empty: %s", img_path)
            st.warning(f"⚠️ Chart file was created but is empty for '{title}'.")
            try:
                _restore_layout(fig, prev_template, prev_paper_bg, prev_plot_bg, prev_font, prev_xaxis, prev_yaxis)
            except Exception:
                pass
            return None

        # Restore original layout (only the pieces we changed)
        try:
            _restore_layout(fig, prev_template, prev_paper_bg, prev_plot_bg, prev_font, prev_xaxis, prev_yaxis)
        except Exception as e:
            logger.warning("Could not restore full layout for '%s': %s", title, e)

        # All good
        logger.info("Saved chart '%s' → %s", title, img_path)
        return img_path

    except Exception as e:
        logger.exception("Unexpected error in save_chart_image: %s", e)
        st.warning(f"⚠️ Could not save chart '{title}': {e}")
        return None


def _restore_layout(fig, template, paper_bg, plot_bg, font, xaxis_json, yaxis_json):
    """
    Restore the handful of layout properties we changed.
    We avoid blindly replacing the entire layout object to preserve user customizations
    that other parts of the app may rely on.
    """
    try:
        # template
        if template is not None:
            try:
                fig.layout.template = template
            except Exception:
                # as a fallback, update template property if settable
                try:
                    fig.update_layout(template=template)
                except Exception:
                    pass

        # paper / plot bg
        if paper_bg is not None:
            try:
                fig.update_layout(paper_bgcolor=paper_bg)
            except Exception:
                pass
        if plot_bg is not None:
            try:
                fig.update_layout(plot_bgcolor=plot_bg)
            except Exception:
                pass

        # font
        if font is not None:
            try:
                fig.update_layout(font=font)
            except Exception:
                pass

        # restore axis minimal json if present (keeps ticks/labels)
        if xaxis_json:
            try:
                fig.update_xaxes(**xaxis_json)
            except Exception:
                # JSON may contain nested props — ignore if can't set
                pass
        if yaxis_json:
            try:
                fig.update_yaxes(**yaxis_json)
            except Exception:
                pass

    except Exception as e:
        logger.debug("Layout restore encountered non-fatal issue: %s", e)
        # don't rethrow; restore is best effort


def ensure_chart_saved(fig, title: str, saver_func: Callable = save_chart_image, **kwargs) -> Optional[str]:
    """
    Convenience wrapper that tries to save and retries once on failure.
    Returns path or None.
    """
    path = saver_func(title, fig, **kwargs)
    if path:
        return path

    # one retry with a small backoff — often helps with transient kaleido hiccups
    time.sleep(0.35)
    return saver_func(title, fig, **kwargs)


def safe_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a pandas categorical column to string before PDF serialization or before setitem
    operations that may raise 'Cannot setitem on a Categorical with a new category'.
    Use this in the PDF pipeline just before passing dataframes to ReportLab.
    """
    if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
        df = df.copy()
        df[col] = df[col].astype(str)
    return df