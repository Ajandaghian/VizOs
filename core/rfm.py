"""
This module contains functions and classes related to RFM (Recency, Frequency, Monetary) analysis.
"""
from typing import List, Dict, Any
from pandas import DataFrame
import numpy as np
import pandas as pd
from utils.helper_func import calendar_generator, rfm_custom_qcut



#https://chatgpt.com/c/69179db2-bed4-8328-b4d4-d3ec4f74fd69

# class RFMAnalyzer:
#     - __init__(df, date_col, customer_col, amount_col)

#     - make_periods(freq)                # ساخت بازه‌ها: M, Q, Y
#     - calculate_rfm_per_period()        # محاسبه Recency, Frequency, Monetary
#     - score_rfm(method="quantile")      # امتیازدهی RFM
#     - assign_segments(segmentation="rfm_score")   # ساخت سگمنت نهایی
#     - build_rfm_history()               # خروجی نهایی: جدول کامل تاریخچه RFM
#     - transition_matrix()               # ماتریس ترنزیشن

# start = df["order_date"].min().normalize()
# end   = df["order_date"].max().normalize()


class RFMAnalyzer:
    """
    A class to perform RFM analysis on customer data.
    """

    def __init__(self, data: DataFrame, type: str, freq: str = "all"):
        """
        Initializes the RFMAnalyzer with customer data.

        :param data: A pandas DataFrame containing customer transaction data.
        :param type: Type of RFM analysis ('classic', 'weighted', 'kmeans').
        :param freq: Frequency for period grouping ('all', 'M', 'Q', 'Y').
        """
        self.data = data.copy()
        self.start = data["order_date"].min().normalize()
        self.end   = data["order_date"].max().normalize()

        if type not in ['classic', 'weighted', 'kmeans',]:
            raise ValueError("Type must be one of 'classic', 'weighted', or 'kmeans'")
        self.type = type
        if freq not in ['all', 'M', 'Q', 'Y']:
            raise ValueError("freq must be one of 'all', 'M', 'Q', or 'Y'")
        self.freq = freq
        self.rfm = pd.DataFrame()

    def _make_periods(self):
        # ---- Choose period grouping ----
        if self.freq == "all":
            self.data["period"] = "all"
            self.period_end = pd.DataFrame({"period": ["all"], "period_end": [self.end]})

        elif self.freq in ["M", "Q", "Y"]:
            self.data["period"] = self.data["order_date"].dt.to_period(self.freq)
            self.period_end = (
                calendar_generator(self.start, self.end).groupby(self.freq)["date"]
                .max()
                .rename("period_end")
                .reset_index()
                .rename(columns={self.freq: "period"})
            )
        else:
            raise ValueError("freq must be 'all', 'M', 'Q', or 'Y'")

    def _compute_rfm_values(self):
        """
        Computes RFM metrics for all types of analysis.
        """
        self._make_periods()
        period_end = self.period_end
        # ---- Aggregate: customer x period ----
        self.rfm = self.data.groupby(["customer_id", "period"]).agg(
            last_order=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("sales_amount", "sum")
        ).reset_index()

        # ---- Attach correct period_end ----
        self.rfm = self.rfm.merge(period_end, on="period", how="left")

        # ---- Recency calculation (always correct) ----
        self.rfm["recency"] = (self.rfm["period_end"] - self.rfm["last_order"]).dt.days

        return self.rfm

    def assign_scores(self):
        if self.type == 'classic':
            # ---- Scoring inside each period ----
            if self.freq == "all":
                # One single group
                self.rfm["recency_score"]   = rfm_custom_qcut(self.rfm["recency"],  labels=[5,4,3,2,1])
                self.rfm["frequency_score"] = rfm_custom_qcut(self.rfm["frequency"],  labels=[1,2,3,4,5])
                self.rfm["monetary_score"]  = rfm_custom_qcut(self.rfm["monetary"],  labels=[1,2,3,4,5])
            else:
                # Separate scoring for each period
                self.rfm["recency_score"] = (
                    self.rfm.groupby("period")["recency"]
                    .transform(lambda x: rfm_custom_qcut(x,labels=[5,4,3,2,1]))
                )
                self.rfm["frequency_score"] = (
                    self.rfm.groupby("period")["frequency"]
                    .transform(lambda x: rfm_custom_qcut(x, labels=[1,2,3,4,5]))
                )
                self.rfm["monetary_score"] = (
                    self.rfm.groupby("period")["monetary"]
                    .transform(lambda x: rfm_custom_qcut(x, labels=[1,2,3,4,5]))
                )

            # ---- 8. Final RFM Score ----
            self.rfm["RFM_Score"] = (
                self.rfm["recency_score"].astype(str)
                + self.rfm["frequency_score"].astype(str)
                + self.rfm["monetary_score"].astype(str)
            )

    def assign_segments(self):
        if self.type == 'classic':
            # 1. Define segments and their conditions in a dictionary
            # The order of definition matters and determines the priority.
            segment_conditions = {
                'Champions': (self.rfm["recency"] >= 4) & (self.rfm['frequency'] >= 4) & (self.rfm['monetary'] >= 4),
                'Loyal Customers': (self.rfm['recency'] >= 3) & (self.rfm['frequency'] >= 3) & (self.rfm['monetary'] >= 3),
                'Potential Loyalists': (self.rfm['recency'] >= 3) & (self.rfm['frequency'] >= 2) & (self.rfm['monetary'] >= 2),
                'New Customers': (self.rfm['recency'] >= 4) & (self.rfm['frequency'] == 1),
                'At Risk': (self.rfm['recency'] <= 2) & (self.rfm['frequency'] >= 3) & (self.rfm['monetary'] >= 3),
                'Customers Needing Attention': (self.rfm['recency'] >= 2) & (self.rfm['recency'] <= 3) &
                                                (self.rfm['frequency'] >= 2) & (self.rfm['frequency'] <= 3) &
                                                (self.rfm['monetary'] >= 2) & (self.rfm['monetary'] <= 3),
                'About to Sleep': (self.rfm['recency'] >= 2) & (self.rfm['recency'] <= 3) & (self.rfm['frequency'] <= 2),
                'Lost': (self.rfm['recency'] <= 2) & (self.rfm['frequency'] <= 2)
            }

            # 3. Apply the rules using np.select
            self.rfm['Segment'] = np.select(list(segment_conditions.values()), list(segment_conditions.keys()), default='Others')

        return self.rfm

    def run(self):
        self._compute_rfm_values()
        self.assign_scores()
        self.assign_segments()

        self.rfm.drop(columns=[
                'last_order', 'period_end',
                'recency', 'frequency', 'monetary',
                'recency_score', 'frequency_score', 'monetary_score'
                ], inplace=True
        )

        return self.rfm

def rfm_analysis(data: DataFrame, type: str, freq: str = "all") -> DataFrame:
    """
    Performs RFM analysis on the provided customer data.
    :param data: A pandas DataFrame containing customer transaction data.
    :param type: Type of RFM analysis ('classic', 'weighted', 'kmeans').
    :param freq: Frequency for period grouping ('all', 'M', 'Q', 'Y').
    :return: A pandas DataFrame with RFM scores for each customer.
    """
    analyzer = RFMAnalyzer(data=data, type=type, freq=freq)

    return analyzer.run()

if __name__ == "__main__":
    from utils.data_loader import DataLoader
    data = DataLoader().sample_data()
    rfm_result = rfm_analysis(data=data, type='classic', freq='all')
    print(rfm_result.head())
