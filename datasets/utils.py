import pandas as pd


def x_y_split(
    df: pd.DataFrame,
    y_col: str
):
    return df.drop([y_col], axis=1), df[y_col]
