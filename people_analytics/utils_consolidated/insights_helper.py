# utils_consolidated/insights_helper.py
"""
Small helper that consolidates insights lists into readable sentences.
"""

def flatten_insights(insights):
    """
    insights: list of strings (or lists)
    returns single readable string
    """
    if not insights:
        return ""
    flat = []
    for it in insights:
        if isinstance(it, (list, tuple)):
            flat.extend([str(x) for x in it])
        else:
            flat.append(str(it))
    # join with bullet points
    return " • ".join(flat)