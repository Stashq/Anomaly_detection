import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from .neural_net import NeuralNet


class CNNClassifier(NeuralNet):
    def __init__(
        self,
        lr: float = 1e-4,
        Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(
            Loss_fn=torch.nn.CrossEntropyLoss,
            device=device
        )
        self._init_layers()
        self.optimizer = Optimizer(self.parameters(), lr=lr)
        self.to(device)

    def _init_layers(self):
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5,
                               stride=1, padding=0, dilation=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5,
                               stride=1, padding=0, dilation=1)
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def _run_minibatch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        purpose: str,
        loss_sum: torch.Tensor,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
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
        y_trues = torch.cat((y_trues, y), 0)
        y_preds = torch.cat((y_preds, torch.argmax(y_pred, axis=1)), 0)
        return loss_sum, y_trues, y_preds
