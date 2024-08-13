import pandas as pd
from pandas import DataFrame


def load_ashfakyeafi() -> DataFrame:
    """https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification"""

    df = pd.read_csv("./datasets/data/ashfakyeafi.csv")
    # remote line "{"mode":"full","isActive":false}"
    df.drop(df[df["Category"] == '{"mode":"full"'].index, inplace=True)
    return df
