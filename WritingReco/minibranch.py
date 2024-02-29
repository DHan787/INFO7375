import numpy as np


class InputNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def partial_fit(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        else:
            batch_mean = np.mean(X, axis=0)
            batch_std = np.std(X, axis=0)
            n_total_samples = X.shape[0] + len(X)
            momentum = X.shape[0] / n_total_samples
            self.mean = momentum * self.mean + (1 - momentum) * batch_mean
            self.std = momentum * self.std + (1 - momentum) * batch_std

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Fit the normalizer first by calling the partial_fit method.")

        X_normalized = (X - self.mean) / self.std
        return X_normalized


