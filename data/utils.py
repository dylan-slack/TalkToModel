from sklearn.base import BaseEstimator, TransformerMixin


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X.astype(object)
        except AttributeError:  # found tensor
            pass
