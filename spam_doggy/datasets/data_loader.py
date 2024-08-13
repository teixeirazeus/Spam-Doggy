import pandas as pd
from pandas import DataFrame
from pathlib import Path


def load_ashfakyeafi() -> DataFrame:
    """https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification"""
    
    module_dir = Path(__file__).parent
    file_path = module_dir / "data/ashfakyeafi.csv"

    df = pd.read_csv(file_path)
    # remote line "{"mode":"full","isActive":false}"
    df.drop(df[df["Category"] == '{"mode":"full"'].index, inplace=True)
    return df
