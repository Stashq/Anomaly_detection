import pandas as pd
import seaborn as sns
from typing import Union
from .utils import x_y_split

sns.set_style("darkgrid")
dir_path = "data/UCRArchive_2018"


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


def get_dataset(
    ds: str,
    dir_path: str = dir_path,
    ext: str = "tsv",
    sep: str = '\t',
    train_test_spliting: bool = True,
    x_y_spliting: bool = False,
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
    if x_y_spliting:
        if y_col is None:
            raise ValueError("\"y_col\" must be defined")
        if train_test_spliting:
            X_train, y_train = x_y_split(train, y_col)
            X_test, y_test = x_y_split(test, y_col)
            result = ((X_train, y_train), (X_test, y_test))
        else:
            result = pd.concat([train, test])
            result = x_y_split(result, y_col)
    else:
        if train_test_spliting:
            result = train, test
        else:
            result = pd.concat([train, test])
    return result
