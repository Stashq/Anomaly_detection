import torch
import torch.nn as nn
from typing import Type
from src.neural_net import NeuralNet
from typing import Literal
from torch.utils.tensorboard import SummaryWriter


class LSTM_AD(NeuralNet):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_next_predicted: int,
        hidden_size1: int,
        hidden_size2: int,
        n_layers: int = 1,
        dropout: float = None,
        lr: float = 1e-4,
        Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(
            Loss_fn=torch.nn.MSELoss,
            device=device
        )
        self.input_size = input_size
        self.output_size = output_size
        self.n_next_predicted = n_next_predicted
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.n_layers = n_layers
        self.dropout = dropout
        self._init_layers()
        self.optimizer = Optimizer(self.parameters(), lr=lr)
        self.to(device)

    def _init_layers(
        self,
        input_size: int,
        output_size: int,
        n_next_predicted: int,
        hidden_size1: int,
        hidden_size2: int,
        n_layers: int = 1,
    ):
        self.input_lstms = [
            nn.LSTM(
                input_size=1,
                hidden_size=hidden_size1,
                num_layers=n_layers,
                # dropout=self.dropout,
                batch_first=True,
            )
            for _ in range(input_size)
        ]
        self.output_lstm = [
            [
                nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size1,
                    num_layers=n_layers,
                    # dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(output_size)
            ]
        ]
        nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.seq_len,
            num_layers=self.n_layers,
            # dropout=self.dropout,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.seq_len,
            num_layers=self.n_layers,
            # dropout=drop_prob,
            batch_first=True,
        )
        self.mlp_decoder = torch.nn.Linear(
            in_features=self.seq_len,
            out_features=1,
        )

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
        data_storage: Literal["records", "single"],
        validation_loader: torch.utils.data.DataLoader = None,
        epochs: int = 20,
        model_name: str = "1",
        logdir: str = "logs",
        flush: bool = False,
        verbose: int = 0
    ):
        '''Train model on given dataset. Optionaly validate training progress
        on validation dataset. Write loss and score while executing.'''
        writer = SummaryWriter(logdir + "/" + model_name)
        for epoch in range(epochs):
            self._run(train_loader, epoch, data_storage,
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
        data_storage: Literal["records", "single"],
        purpose: Literal["train", "validation"] = None,
        writer: SummaryWriter = None,
        verbose: int = 0
    ):
        '''Run model on given dataset. After that, write loss and score'''
        if purpose is None:
            print("Warning: no purpose of \"_run\" given. Nothing happened.")
            return None
        loss_sum = 0
        x_preds = torch.Tensor([]).to(self.device)
        if data_storage == "records":
            for x, y, anomalies in data_loader:
                loss_sum, x_preds = self._run_minibatch(
                    x=x, purpose=purpose,
                    loss_sum=loss_sum, x_preds=x_preds
                )
        elif data_storage == "single":
            raise NotImplementedError()
            x, y = data_loader.dataset[:]
            for i in range(len(data_loader.dataset) - self.seq_len):
                # x_batch, y_ = self.get_batch(data_loader)
                x_batch = x[i:i+self.seq_len].unsqueeze(0)
                loss_sum, x_preds = self._run_minibatch(
                    x=x_batch, purpose=purpose,
                    loss_sum=loss_sum, x_preds=x_preds
                )

        x_trues, _, _ = data_loader.dataset[:]
        self._write_epoch_summary(
            loss=loss_sum/len(data_loader),
            score=self.loss_fn(x_trues, x_preds),
            score_name="MSE",
            epoch=epoch,
            purpose=purpose,
            verbose=verbose,
            writer=writer
        )

    def _run_minibatch(
        self,
        x: torch.Tensor,
        purpose: str,
        loss_sum: torch.Tensor,
        x_preds: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        '''- Pass minibatch through the model,
        - apply backpropagation if training is the purpose,
        - adds the currently produced loss,
        - append true y from minibatch,
        - append pred y predicted for minibatch.
        '''
        x_pred = self(x)
        x_preds = torch.cat((x_preds, x_pred), 0)

        loss = self.loss_fn(x_pred, x)
        loss_sum += loss.item()

        if purpose == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_sum, x_preds

    def forward(self, x: torch.Tensor):
        '''
        Parameters
        -----------
        x: torch.Tensor, size: (batch_size, seq_len, input_size)

        Returns
        -----------
        torch.Tensor - size: (batch_size, seq_len, input_size)'''
        batch_size = x.size()[0]
        hn = self._init_state(batch_size=batch_size, h_size=self.seq_len)
        cn = self._init_state(batch_size=batch_size, h_size=self.seq_len)
        emb, (hn, cn) = self.lstm_encoder(x, (hn, cn))

        # we pass hn (and cn?) to decoder
        # input of first from last LSTM cell is last in sequence x
        # we use linear layer to predict x' from every LSTM cell
        # x_(i)' is used as input of next LSTM cell to predict x_(i-1)
        x_preds = torch.empty(size=x.size()).to(self.device)
        x_pred = emb[:, -1, :].unsqueeze(1)
        for i in reversed(range(self.seq_len)):
            if i < self.seq_len - 1:
                x_pred = x_pred.unsqueeze(1)
                x_pred, (hn, cn) = self.lstm_decoder(x_pred, (hn, cn))
            x_pred = x_pred.squeeze(1)
            x_pred = self.mlp_decoder(x_pred)
            x_preds[:, i, :] = x_pred
        return x_preds