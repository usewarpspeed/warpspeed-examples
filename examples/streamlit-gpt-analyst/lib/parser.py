import pandas as pd

def to_dataframe(raw_data: str) -> pd.DataFrame:
    return pd.read_json(raw_data)