import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

sns.set_style("darkgrid")
dir_path = "datasets/UCRArchive_2018"


def get_path(
    ds: str,
    dir_path: str = dir_path,
    ext: str = "tsv",
    train: bool = True
):
    path = "{0}/{1}/{1}".format(dir_path, ds)
    if train:
        path += "_TRAIN"
    else:
        path += "_TEST"
    if ext:
        path += "." + ext
    return path


def x_y_split(
    df: pd.DataFrame,
    y_col: str
):
    return df.drop([y_col], axis=1), df[y_col]


def get_dataset(
    ds: str,
    dir_path: str = dir_path,
    ext: str = "tsv",
    sep: str = '\t',
    train_test_split: bool = True,
    x_y_split: bool = False,
    y_col: Union[str, int] = None
):
    train = pd.read_csv(
        get_path(ds, dir_path=dir_path, train=True),
        sep=sep,
        header=None
        )
    test = pd.read_csv(
        get_path(ds, dir_path=dir_path, train=False),
        sep=sep,
        header=None
        )
    if x_y_split:
        if y_col is None:
            raise ValueError("\"y_col\" must be defined")
        if train_test_split:
            X_train, y_train = x_y_split(train, y_col)
            X_test, y_test = x_y_split(test, y_col)
            result = ((X_train, y_train), (X_test, y_test))
        else:
            result = pd.concat([train, test])
            result = x_y_split(result, y_col)
    else:
        if train_test_split:
            result = train, test
        else:
            result = pd.concat([train, test])
    return result
