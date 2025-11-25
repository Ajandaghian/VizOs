"""
Utility helper functions for various tasks.
"""
import pandas as pd
import numpy as np

def calendar_generator(start, end):
    calendar = pd.date_range(start=start, end=end, freq="D").to_frame(name="date")

    # ---- Generate period columns ----
    calendar["M"] = calendar["date"].dt.to_period("M")
    calendar["Q"] = calendar["date"].dt.to_period("Q")
    calendar["Y"] = calendar["date"].dt.to_period("Y")

    return calendar

def rfm_custom_qcut(col: str ,series: pd.Series):
    """
    Custom quantile cut function for RFM scoring.
    Assigns scores based on quantiles, handling edge cases.

    :param col: The RFM metric column name ('recency', 'frequency', or 'monetary').
    :param series: The pandas Series to be binned.
    :return: A pandas Series with assigned scores.

    """
    if col not in ['recency', 'frequency', 'monetary']:
        raise ValueError("Column must be one of 'recency', 'frequency', or 'monetary'")

    if col == 'recency':
        labels = [5,4,3,2,1]
    else:
        labels = [1,2,3,4,5]

    if series.nunique() < 2:
        return pd.Series(
                # [labels[len(labels)//2]]
                3 * len(series),
                index=series.index)

    elif series.nunique() < len(labels): # between 2 and 4 unique values
        # Create bins based on unique values
        unique_vals = series.unique()
        bins = [0] + sorted(unique_vals.tolist()) + [max(unique_vals) + 1]

        return pd.cut(series, bins=bins, labels=labels[:len(bins)-1], include_lowest=True)

    else:
        return pd.qcut(series, len(labels), labels=labels[:pd.qcut(series, len(labels), duplicates='drop').cat.categories.size], duplicates='drop')
