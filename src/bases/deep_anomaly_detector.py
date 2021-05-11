from .neural_net import NeuralNet
from .gausian_anomaly_detector import GaussianAnomalyDetector

from typing import Type, Literal
import torch
from torch.utils.tensorboard import SummaryWriter


class DeepAnomalyDetector(NeuralNet):
    def __init__(
        self,
        Loss_fn: torch.nn.modules.module.Module,
        prob_threshold: float = 1e-3,
        lr: float = 1e-4,
        Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(
            Loss_fn=torch.nn.MSELoss,
            device=device
        )
        self.anomaly_detector = GaussianAnomalyDetector(
            threshold=prob_threshold
        )

    def train_detector(
        self,
        train_loader: torch.utils.data.DataLoader,
        verbose: int = 1
    ):
        errors = self._run(
            data_loader=train_loader,
            epoch=0,
            purpose="train detector",
            writer=None,
            return_error=True,
            verbose=verbose
        )
        self.anomaly_detector.fit(errors)

    def predict_anomalies(self, x, y, indicies: bool = True):
        y_pred = self(x)
        error = torch.nn.MSELoss(y_pred, y)
        pred_anomalies = self.anomaly_detector.predict_anomaly(
                         error, indices=indicies)
        return pred_anomalies

    def evaluate_detector(
        self,
        validation_loader: torch.utils.data.DataLoader,
        verbose: int = 0
    ):
        errors = self._run(
            data_loader=validation_loader,
            epoch=0,
            purpose="evaluate detector",
            writer=None,
            return_error=True,
            verbose=verbose
        )
        true_anomalies = validation_loader.dataset[:][2]\
            .squeeze(1).cpu().numpy()
        pred_anomalies = self.anomaly_detector\
            .predict_anomaly(errors, indices=False)
        print("\"evaluate_detector\": Here should be some metric... \
            (not implemented)")
        return true_anomalies, pred_anomalies

    def _run(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        data_storage: Literal["records", "single"],
        purpose: Literal["train", "validation"] = None,
        writer: SummaryWriter = None,
        return_error: bool = False,
        verbose: int = 0
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
            return torch.abs(y_trues - y_preds).cpu().detach().numpy()

    def _run_minibatch(
        self,
        x: torch.Tensor,
        purpose: str,
        loss_sum: torch.Tensor,
        preds: torch.Tensor,
        y: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        '''- Pass minibatch through the model,
        - apply backpropagation if training is the purpose,
        - adds the currently produced loss,
        - append true y from minibatch,
        - append pred y predicted for minibatch.
        '''
        pred = self(x)
        if y is None:
            loss = self.loss_fn(pred, x)
            preds = torch.cat([preds, pred], 0)
        else:
            loss = self.loss_fn(pred, y)
            preds = torch.cat([preds, pred])

        if purpose == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_sum += loss.item()
        return loss_sum, preds

    def get_true_pred_anomalies(
        self,
        validation_loader: torch.utils.data.DataLoader,
        verbose: int = 0
    ):
        errors = self._run(
            data_loader=validation_loader,
            epoch=0,
            purpose="evaluate detector",
            writer=None,
            return_error=True,
            verbose=verbose
        )

        pred_anomalies = self.anomaly_detector\
            .predict_anomaly(errors, indices=False)
        true_anomalies = validation_loader.dataset[:][2]\
            .squeeze(1).cpu().numpy()
        return true_anomalies, pred_anomalies
