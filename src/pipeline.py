from .bases.deep_anomaly_detector import DeepAnomalyDetector
from .utils import data
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from typing import Tuple


def train_evaluate(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    test_df: pd.DataFrame,
    window_size: int,
    AnomalyDetector: DeepAnomalyDetector,
    model_name: str,
    n_epochs: int = 20,
    verbose: int = 1,
    plot_results: bool = True,
    device: str = "cpu",
    **params
):
    model = AnomalyDetector(
            window_size=window_size,
            lr=1e-4,
            Optimizer=torch.optim.Adam,
            device=device,
            **params
    ).to(device)
    model.train(train_loader=train_loader, validation_loader=None,
                n_epochs=n_epochs, model_name=model_name, logdir="logs",
                verbose=1)
    model.train_detector(train_loader)
    if plot_results:
        show_results(
            model=model,
            validation_df=test_df,
            validation_loader=test_loader,
            window_size=window_size
        )


def create_sin_dataset(
    window_size: int,
    device: str = 'cpu',
    shuffle: bool = False,
    return_val_df: bool = True,
    model_type: str = "CNN"
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
        window_size=window_size,
        model_type=model_type,
        device=device,
        anomalies=anomalies,
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
        window_size=window_size,
        model_type=model_type,
        device=device,
        anomalies=anomalies,
        **params
    )
    if return_val_df:
        return train_loader, test_loader, df
    return train_loader, test_loader


def _plot_with_anomalies(
    df: pd.DataFrame,
    true_anomalies: np.ndarray,
    pred_anomalies: np.ndarray,
    ax: plt.axis,
    window_size: int = 0,
):
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
        pred_anomalies + window_size,
        df.iloc[pred_anomalies + window_size],
        c="green",
        marker="^",
        s=50,
        zorder=11,
        label='predicted anomaly'
    )
    ax.legend()


def _plot_predictions(df, preds, window_size, ax, alpha=0.04):
    if preds.ndim == 1:
        ax.plot(
            df.index[window_size:],
            preds,
            zorder=10
        )
    elif preds.ndim == 2:
        for i in range(len(preds)):
            ax.plot(
                df.index[i:i+window_size],
                preds[i],
                color="orange",
                alpha=alpha,
                zorder=10
            )
    else:
        print("Warning: wrong dimention of \"preds\". Nothing plotted.")


def show_results(
    model: DeepAnomalyDetector,
    validation_df: pd.DataFrame,
    validation_loader: torch.utils.data.DataLoader,
    window_size: int,
    shift: bool = True,
    plot_values: bool = True,
    plot_predictions: bool = True,
    plot_anomalies: bool = True,
    plot_predicted_anomalies: bool = True,
    figsize: Tuple = (10, 6),
    verbose: int = 0
):
    true_anomalies, pred_anomalies =\
        model.get_true_pred_anomalies(
            validation_loader=validation_loader,
            verbose=verbose
        )
    true_anomalies, pred_anomalies =\
        np.where(true_anomalies > 0)[0], np.where(pred_anomalies > 0)[0]

    pred = model(validation_loader.dataset[:][0])\
        .squeeze(-1).cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    if plot_values:
        sns.lineplot(
            data=validation_df,
            x="timestamp",
            y="value",
            ax=ax,
            label="true values"
        )
    if plot_predictions:
        _plot_predictions(validation_df, pred, window_size, ax)

    ymin, ymax = ax.get_ylim()
    if plot_anomalies:
        ax.scatter(
            true_anomalies,
            [ymin] * len(true_anomalies),
            c="red",
            marker="X",
            s=100,
            zorder=11,
            label='true anomaly'
        )
    if plot_predicted_anomalies:
        ax.scatter(
            pred_anomalies + window_size,
            [ymin] * len(pred_anomalies + window_size),
            c="green",
            marker="^",
            s=30,
            zorder=12,
            label='predicted anomaly'
        )

    # handles, labels = ax.get_legend_handles_labels()
    # patch = mpatches.Patch(color='orange', label='[redicted values')
    # handles.append(patch)
    # ax.legend(handles=handles)
    ax.legend()
