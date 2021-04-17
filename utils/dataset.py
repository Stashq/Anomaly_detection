import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_dataset(
    ds: str,
    dir_path: str = dir_path,
    ext: str = "tsv",
    sep: str = '\t'
):
    train_df = pd.read_csv(
        get_path(ds, dir_path=dir_path, train=True),
        sep=sep,
        header=None
        )
    test_df = pd.read_csv(
        get_path(ds, dir_path=dir_path, train=False),
        sep=sep,
        header=None
        )
    return train_df, test_df
