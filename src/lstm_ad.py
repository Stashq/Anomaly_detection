import torch
import torch.nn as nn
from typing import Type
from .bases.deep_anomaly_detector import DeepAnomalyDetector
from typing import Literal
from torch.utils.tensorboard import SummaryWriter


class LSTM_AD(DeepAnomalyDetector):
    def __init__(
        self,
        window_size: int,
        hidden_size1: int,
        hidden_size2: int,
        n_layers: int = 1,
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
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.n_layers = n_layers
        self._init_layers(**params)
        self.optimizer = Optimizer(self.parameters(), lr=lr)
        self.to(device)

    def _init_layers(
        self,
        input_size: int = 1,
        output_size: int = 1,
        fc_size: int = 128,
    ):
        self.input_lstms = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size1,
            num_layers=self.n_layers,
            # dropout=self.dropout,
            batch_first=True,
        )
        self.output_lstms = nn.LSTM(
            input_size=self.hidden_size1,
            hidden_size=self.hidden_size2,
            num_layers=self.n_layers,
            # dropout=self.dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.hidden_size2*self.window_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor):
        '''
        Parameters
        -----------
        x: torch.Tensor, size: (batch_size, seq_len, input_size)

        Returns
        -----------
        torch.Tensor - size: (batch_size, seq_len, input_size)'''
        batch_size = x.size()[0]

        hn = self._init_state(batch_size=batch_size, h_size=self.hidden_size1)
        cn = self._init_state(batch_size=batch_size, h_size=self.hidden_size1)
        x, (hn, cn) = self.input_lstms(x, (hn, cn))

        hn = self._init_state(batch_size=batch_size, h_size=self.hidden_size2)
        cn = self._init_state(batch_size=batch_size, h_size=self.hidden_size2)
        x, (hn, cn) = self.output_lstms(x, (hn, cn))

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _init_state(self, batch_size: int, h_size: int) -> torch.Tensor:
        '''
        Returns
        -----------
        torch.Tensor - size: (n_layers, batch_size, hidden_size)
        '''
        state = torch.randn(
            self.n_layers,
            batch_size,
            h_size,
            device=self.device
        )
        return state

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader = None,
        n_epochs: int = 20,
        model_name: str = "1",
        logdir: str = "logs",
        flush: bool = False,
        train_detector: bool = False,
        verbose: int = 0
    ):
        '''Train model on given dataset. Optionaly validate training progress
        on validation dataset. Write loss and score while executing.'''
        writer = SummaryWriter(logdir + "/" + model_name)
        for epoch in range(n_epochs):
            self._run(train_loader, epoch,
                      "train", writer, verbose)
            if validation_loader is not None:
                with torch.no_grad():
                    self._run(validation_loader, epoch, "validation",
                              writer, verbose)
            if flush is True:
                writer.flush()
        writer.close()

    def _run(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        purpose: Literal["train", "validation"] = None,
        writer: SummaryWriter = None,
        verbose: int = 0,
        return_error: bool = False
    ):
        '''Run model on given dataset. After that, write loss and score'''
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
        if return_error:
            return torch.abs(y_trues - y_preds)\
                .squeeze(-1).cpu().detach().numpy()

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
