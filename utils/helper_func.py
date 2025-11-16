"""
Utility helper functions for various tasks.
"""
import pandas as pd

def calendar_generator(start, end):
    calendar = pd.date_range(start=start, end=end, freq="D").to_frame(name="date")

    # ---- Generate period columns ----
    calendar["M"] = calendar["date"].dt.to_period("M")
    calendar["Q"] = calendar["date"].dt.to_period("Q")
    calendar["Y"] = calendar["date"].dt.to_period("Y")

    return calendar

def rfm_custom_qcut(x, labels):
    if x.nunique() < 2:
        return pd.Series([labels[len(labels)//2]] * len(x), index=x.index)
    else:
        try:
            return pd.qcut(x, len(labels), labels=labels)

        except ValueError:
            return x.rank(method='dense').astype(int)
