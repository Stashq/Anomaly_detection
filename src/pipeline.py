from .bases.deep_anomaly_detector import DeepAnomalyDetector
from .utils import data
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple


def train_evaluate(AnomalyDetector: DeepAnomalyDetector):
    raise NotImplementedError()


def create_sin_dataset(
    window: int,
    device: str = 'cpu',
    shuffle: bool = False,
    return_val_df: bool = True
):
    x = np.sin(np.linspace(0, 32*np.pi, 1024))
    t = np.arange(len(x))
    anomalies = torch.zeros(len(x))

    params = {
        'batch_size': 10,
        'shuffle': shuffle
    }
    train_loader = data.create_data_loader(
        x=torch.tensor(x),
        anomalies=anomalies,
        window=window,
        device=device,
        **params
    )

    x = np.sin(np.linspace(0, 32*np.pi, 1024))
    x[350:400] = 0
    anomalies = torch.zeros(len(x))
    anomalies[350:400] = 1

    t = np.arange(len(x))
    df = pd.DataFrame({"timestamp": t, "value": x}).set_index("timestamp")

    test_loader = data.create_data_loader(
        x=torch.tensor(x),
        anomalies=anomalies,
        window=window,
        device=device,
        **params
    )
    if return_val_df:
        return train_loader, test_loader, df
    return train_loader, test_loader


def plot_with_anomalies(
    df: pd.DataFrame,
    true_anomalies: np.ndarray,
    pred_anomalies: np.ndarray,
    window: int = 0,
    figsize: Tuple = (10, 6)
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df,
        x="timestamp",
        y="value",
        ax=ax
    )
    ax.scatter(
        true_anomalies,
        df.iloc[true_anomalies],
        c="red",
        marker="X",
        s=100,
        zorder=10,
        label='true anomaly'
    )
    ax.scatter(
        pred_anomalies + window,
        df.iloc[pred_anomalies + window],
        c="green",
        marker="^",
        s=50,
        zorder=11,
        label='predicted anomaly'
    )
    ax.legend()
