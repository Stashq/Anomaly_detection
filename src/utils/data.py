import torch
import pandas as pd
from typing import Literal


def x_y_split(
    df: pd.DataFrame,
    y_col: str
):
    return df.drop([y_col], axis=1), df[y_col]


def create_data_loader(
    x: torch.Tensor,
    window: int,
    model_type: Literal["CNN", "LSTM"] = "CNN",
    device: str = 'cpu',
    anomalies: torch.Tensor = None,
    **params
) -> torch.utils.data.DataLoader:
    new_x = torch.tensor([])
    new_y = torch.tensor([])
    new_anomalies = torch.tensor([])

    for i in range(x.shape[0] - window):
        new_x = torch.cat([new_x, x[i:i+window].unsqueeze(0)])
        new_y = torch.cat([new_y, x[i+window].unsqueeze(0)])
        if anomalies is not None:
            new_anomalies = torch.cat([
                new_anomalies,
                anomalies[i:i+window].unsqueeze(0)
            ])  # POWINNA BYĆ JEDNA WARTOŚĆ NA PRÓBKĘ
        else:
            new_anomalies = torch.cat([
                new_anomalies,
                torch.tensor([0]*window).unsqueeze(0)
            ])  # POWINNA BYĆ JEDNA WARTOŚĆ NA PRÓBKĘ

    if model_type == "CNN":
        # (batch_size, number_of_chanels, sequence_length)
        new_x = new_x.unsqueeze(1).float().to(device)
    elif model_type == "LSTM":
        # (batch_size, sequence_length, features_size)
        new_x = new_x.unsqueeze(-1).float().to(device)
    else:
        raise ValueError("Wrong \"model_type\".")
    new_y = new_y.unsqueeze(1).float().to(device)
    new_anomalies = new_anomalies.unsqueeze(1).float().to(device)
    dataset = torch.utils.data.TensorDataset(
        new_x,
        new_y,
        new_anomalies,
    )
    return torch.utils.data.DataLoader(dataset, **params)

# def 
