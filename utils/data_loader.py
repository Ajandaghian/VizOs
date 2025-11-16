"""
Data loading utility functions.
"""
import pandas as pd
import pathlib


class DataLoader:
    """
    A class to load data from various sources.
    """
    def __init__(self):
        # self.source = source
        pass

    def sample_data(self,name: str = 'Superstore') -> pd.DataFrame:
        """
        Load a sample of the data with n rows.

        :return: pandas DataFrame with sampled data.
        """
        sample_paths = {
            'Superstore': '../data/cleaned_sample_superstore.xlsx',
            # 'Online Retail': '../data/cleaned_sample_online_retail.xlsx',
        }
        if name not in sample_paths:
            raise ValueError(f"Sample data '{name}' not found. Available samples: {list(sample_paths.keys())}")
        path = pathlib.Path(__file__).parent/sample_paths[name]
        return pd.read_excel(path)

    #TODO: Enable loading from different sources
    #TODO: Support more file formats
    #TODO: handling the column mapping


    # def load_data(self):
    #     """
    #     Load data from:
    # - a local path (str or pathlib.Path)
    # - a Streamlit UploadedFile object

    # Supports CSV, Excel, Parquet.
    # Returns: pandas DataFrame
    # """

    # # --- 1. If Streamlit uploaded file ---
    # if hasattr(source, "read") and not isinstance(source, (str, pathlib.Path)):
    #     filename = source.name.lower()

    #     if filename.endswith(".csv"):
    #         return pd.read_csv(source)

    #     elif filename.endswith((".xlsx", ".xls")):
    #         return pd.read_excel(source)

    #     elif filename.endswith(".parquet"):
    #         return pd.read_parquet(source)

    #     else:
    #         raise ValueError("Unsupported uploaded file format.")

    # # --- 2. If local path ---
    # path = pathlib.Path(source)
    # if not path.exists():
    #     raise FileNotFoundError(f"File not found: {path}")

    # if path.suffix == ".csv":
    #     return pd.read_csv(path)

    # elif path.suffix in [".xlsx", ".xls"]:
    #     return pd.read_excel(path)

    # elif path.suffix == ".parquet":
    #     return pd.read_parquet(path)

    # else:
    #     raise ValueError("Unsupported file format.")


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.sample_data('Superstore')
    print(df.head())