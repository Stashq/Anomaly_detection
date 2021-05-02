import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Type, Literal


class NeuralNet(nn.Module):
    def __init__(
        self,
        loss_fn: torch.nn.modules.module.Module,
        lr: float = 1e-4,
        Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.loss_fn = loss_fn()
        self.device = device

    def _init_layers(self):
        pass

    def forward(self, x):
        pass

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
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
            self._run(train_loader, epoch, "train", writer, verbose)
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
        verbose: int = 0
    ):
        '''Run model on given dataset. After that, write loss and score'''
        if purpose is None:
            print("Warning: no purpose of \"_run\" given. Nothing happened.")
            return None
        loss_sum = 0
        y_trues = torch.tensor([]).to(self.device)
        y_preds = torch.tensor([]).to(self.device)
        for x, y in data_loader:
            loss_sum, y_trues, y_preds = self._run_minibatch(
                x=x, y=y, purpose=purpose,
                loss_sum=loss_sum, y_trues=y_trues, y_preds=y_preds
            )
        self._write_epoch_summary(
            loss=loss_sum/len(data_loader),
            score=self.f1_loss(y_trues, y_preds),
            score_name="F1-score",
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

    def _write_epoch_summary(
        self,
        loss: torch.Tensor,
        score: torch.Tensor,
        epoch: int,
        purpose: str,
        score_name: str = "F1-score",
        verbose: int = 1,
        writer: torch.utils.tensorboard.SummaryWriter = None
    ):
        '''Print loss and score or write them to tensorboard.'''
        if writer is None or verbose > 0:
            print("Epoch %d, dataset %s: Loss - %f, %s - %f" %
                  (epoch, purpose, loss, score_name, score))
        if writer is not None:
            writer.add_scalar(("Loss/" + purpose), loss, epoch)
            writer.add_scalar((score_name + "/" + purpose), score, epoch)

    def f1_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        is_training: bool = False
    ) -> torch.Tensor:
        '''Calculate F1 score. Can work with gpu tensors

        The original implmentation is written by Michal Haltuf on Kaggle.

        Returns
        -------
        torch.Tensor
            `ndim` == 1. 0 <= val <= 1

        Reference
        ---------
        - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score\
            .metric
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics\
            .f1_score.html#sklearn.metrics.f1_score
        - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1\
            -score-in-case-of-multi-label-classification/28265/6
        '''
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)

        tp = (y_true * y_pred).sum().to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        f1.requires_grad = is_training
        return f1
