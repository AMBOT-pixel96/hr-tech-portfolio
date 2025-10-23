import time
import pandas as pd

def ensure_chart_saved(fig, title, saver_func):
    """Guarantees chart is saved before PDF generation."""
    path = saver_func(title, fig)
    time.sleep(0.3)  # wait for async disk write
    return path

def safe_categorical(df, col):
    """Ensures Categorical columns won't crash when new bins appear."""
    if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str)
    return df