import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from .neural_net import NeuralNet
from typing import Literal
from torch.utils.tensorboard import SummaryWriter


class DeepAnt(NeuralNet):
    def __init__(
        self,
        window_size: int,
        lr: float = 1e-4,
        Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu"),
        **params
    ):
        super().__init__(
            Loss_fn=torch.nn.MSELoss,
            device=device
        )
        self.window_size = window_size
        self._init_layers(**params)
        self.optimizer = Optimizer(self.parameters(), lr=lr)
        self.to(device)

    def _init_layers(
        self,
        out_channels_1: int = 3,
        kernel_size_1: int = 5,
        stride_1: int = 1,
        padding_1: int = 0,
        out_channels_2: int = 3,
        kernel_size_2: int = 5,
        stride_2: int = 1,
        padding_2: int = 0,
        max_pool: int = 2,
        linear_out_1: int = 128,
    ):
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels_1,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            kernel_size=kernel_size_2,
            stride=stride_2,
            padding=padding_2
        )
        self.pool = nn.MaxPool1d(max_pool)
        linear_in = int((self.window_size - kernel_size_1 + 1)/max_pool)
        linear_in = int((linear_in - kernel_size_2 + 1)/max_pool)
        linear_in *= out_channels_2
        self.fc1 = nn.Linear(linear_in, linear_out_1)
        self.fc2 = nn.Linear(linear_out_1, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _run(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        purpose: Literal["train", "validation"] = None,
        writer: SummaryWriter = None,
        verbose: int = 0
    ):
        '''Run model on given dataset. After that, write loss and score.
        '''
        if purpose is None:
            print("Warning: no \"purpose\" of \"_run\" given. \
                Nothing happened.")
            return None
        loss_sum = 0
        y_preds = torch.tensor([]).to(self.device)
        for x, y, anomalies in data_loader:
            loss_sum, y_preds = self._run_minibatch(
                x=x, y=y, purpose=purpose,
                loss_sum=loss_sum, y_preds=y_preds
            )
        y_trues = data_loader.dataset.tensors[1]
        self._write_epoch_summary(
            loss=loss_sum/len(data_loader),
            score=self.loss_fn(y_trues, y_preds),
            score_name="MSE",
            epoch=epoch,
            purpose=purpose,
            verbose=verbose,
            writer=writer
        )

    def _run_minibatch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        purpose: str,
        loss_sum: torch.Tensor,
        y_preds: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        '''- Pass minibatch through the model,
        - apply backpropagation if training is the purpose,
        - adds the currently produced loss,
        - append true y from minibatch,
        - append pred y predicted for minibatch.
        '''
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        if purpose == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss_sum += loss.item()
        y_preds = torch.cat([
            y_preds,
            y_pred
        ])
        return loss_sum, y_preds
