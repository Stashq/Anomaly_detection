from scipy.stats import multivariate_normal
import numpy as np


class GaussianAnomalyDetector:
    def __init__(self, threshold: float = 1e-3):
        self.loc = None
        self.scale = None
        self.mvnormal = None
        self.threshold = threshold

    def fit(self, error_vectors):
        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)
        self.mvnormal = multivariate_normal(
            self.mean, self.cov, allow_singular=True
        )

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def predict_proba(self, errors):
        return self.mvnormal.pdf(errors)

    def predict_anomaly(self, errors, indices: bool = True):
        probs = self.predict_proba(errors)
        if indices is True:
            return np.where(probs < self.threshold)
        else:
            results = probs < self.threshold
            return results.astype(int)
