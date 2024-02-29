import numpy as np


class InputNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):

        if self.mean is None or self.std is None:
            raise ValueError("Fit the normalizer first by calling the fit method.")

        X_normalized = (X - self.mean) / self.std
        return X_normalized

    def fit_transform(self, X):

        self.fit(X)
        X_normalized = self.transform(X)
        return X_normalized


