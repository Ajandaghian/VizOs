"""
This module contains functions and classes related to RFM (Recency, Frequency, Monetary) analysis.
"""
from typing import List, Dict, Any
from pandas import DataFrame
import numpy as np
import pandas as pd
from utils.helper_func import calendar_generator, rfm_custom_qcut




AVAILABLE_TYPES = ['classic', 'weighted', 'kmeans',]
AVAILABLE_FREQS = ['all', 'M', 'Q', 'Y']
SEGMENTS_MAP = {
    (4.5, 5.0): 'Champions',
    (4.0, 4.49): 'Loyal',
    (3.0, 3.99): 'Potential Loyalists',
    (2.5, 2.99): 'Promising',
    (2.0, 2.49): 'Needs Attention',
    (1.5, 1.99): 'At Risk',
    (1.0, 1.49): 'Lost'
}


class RFMAnalyzer:
    """
    A class to perform RFM analysis on customer data.
    """

    def __init__(self, data: DataFrame, type: str = 'classic', freq: str = "all", weights: list = [0.33, 0.33, 0.34]):
        f"""
        Initializes the RFMAnalyzer with customer data.

        :param data: A pandas DataFrame containing customer transaction data.
        :param type: Type of RFM analysis {AVAILABLE_TYPES}.
        :param freq: Frequency for period grouping {AVAILABLE_FREQS}.
        :param weights: Optional weights for weighted RFM analysis. in the order of (R, F, M). between 0 and 1. Sum must be 1.
        """

        self.data = data.copy()
        self.start = data["order_date"].min().normalize()
        self.end   = data["order_date"].max().normalize()

        if type not in AVAILABLE_TYPES:
            raise ValueError(f"Type must be one of {AVAILABLE_TYPES}")
        self.type = type

        if freq not in AVAILABLE_FREQS:
            raise ValueError(f"freq must be one of {AVAILABLE_FREQS}")
        self.freq = freq

        if self.type == 'weighted':
            if not weights:
                raise ValueError("Weights must be provided for weighted RFM analysis.")
            if len(weights) !=3:
                raise ValueError("Weights must be a list of three values for R, F, and M respectively.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
        self.weights = weights

        self.rfm = pd.DataFrame()

    def _make_periods(self):
        # ---- Choose period grouping ----
        if self.freq == "all":
            self.data["period"] = "all"
            self.period_end = pd.DataFrame({"period": ["all"], "period_end": [self.end]})

    def _compute_rfm_values(self):
        """
        Computes RFM metrics for all types of analysis.
        """
        #--- Create periods ----
        self._make_periods()

        # ---- Aggregate: customer x period ----
        self.rfm = self.data.groupby(["customer_id", "period"]).agg(
            last_order=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("sales_amount", "sum")
        ).reset_index()

        # ---- Attach correct period_end ----
        self.rfm = self.rfm.merge(self.period_end, on="period", how="left")

        # ---- Recency calculation (always correct) ----
        self.rfm["recency"] = (self.rfm["period_end"] - self.rfm["last_order"]).dt.days

        return self.rfm

    def _calculate_rfm_scores(self):

        # ---- Classic RFM Scoring ----
        self.rfm['R_score'] = rfm_custom_qcut('recency', self.rfm['recency']).astype(int)
        self.rfm['F_score'] = rfm_custom_qcut('frequency', self.rfm['frequency']).astype(int)
        self.rfm['M_score'] = rfm_custom_qcut('monetary', self.rfm['monetary']).astype(int)

        self.rfm['RFM_label'] = (
            self.rfm['R_score'].astype(str) +
            self.rfm['F_score'].astype(str) +
            self.rfm['M_score'].astype(str)
        )

        if self.type == 'classic':
            self.rfm['RFM_score'] = round(self.rfm[['R_score', 'F_score', 'M_score']].mean(axis=1), 1)

        elif self.type == 'weighted':
            R_weight, F_weight, M_weight = self.weights
            self.rfm['RFM_score'] = round(
                self.rfm['R_score'] * R_weight +
                self.rfm['F_score'] * F_weight +
                self.rfm['M_score'] * M_weight,
                1
            )

    def _assign_segments(self):

        def map_segment(score):
            for (low, high), segment in SEGMENTS_MAP.items():
                if low <= score <= high:
                    return segment
            return 'Unknown'

        self.rfm['Segment'] = self.rfm['RFM_score'].apply(map_segment)

        #---- Final RFM DataFrame with clean columns ----
        self.rfm = self.rfm[['customer_id', 'period', 'RFM_score', 'Segment']]

        return self.rfm

    def run(self):
        """
        Runs the complete RFM analysis pipeline.
        """
        self._compute_rfm_values()
        self._calculate_rfm_scores()
        self._assign_segments()

        return self.rfm



def rfm_analysis(data: DataFrame, type: str = 'classic', freq: str = "all", weights: list = None) -> DataFrame:
    """
    A convenience function to perform RFM analysis.

    :param data: A pandas DataFrame containing customer transaction data.
    :param type: Type of RFM analysis {AVAILABLE_TYPES}.
    :param freq: Frequency for period grouping {AVAILABLE_FREQS}.
    :return: A pandas DataFrame with RFM scores and segments.
    """
    rfm_analyzer = RFMAnalyzer(data, type=type, freq=freq, weights=weights if weights else [])
    rfm_result = rfm_analyzer.run()

    return rfm_result



if __name__ == "__main__":

    from utils.data_loader import DataLoader
    data = DataLoader().sample_data()
    print(data.head())
    print('---'*20)

    rfm_class = RFMAnalyzer(data, type='classic', freq='all')
    rfm_final = rfm_class.run()

    print(rfm_final.head())
    print('---'*20)

    print('Weighted RFM Analysis:')
    weights = [0.5, 0.3, 0.2]
    rfm_weighted = RFMAnalyzer(data, type='weighted', freq='all', weights=weights)
    rfm_final_weighted = rfm_weighted.run()

    print(rfm_final_weighted.head())
    print('---'*20)




