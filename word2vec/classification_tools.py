from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import gensim


class ProteinTokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenization step : Transforms list of sequences like 'AAXLMDKSQL...DAB'
    to ['AAX', 'LMD', 'KSQ', ...]

    sklearn-compatible
    """
    def __init__(self, token_size):
        self.token_size = token_size
        pass

    def transform(self, X, y=None):
        X_transformed = []
        for seq in X:
            seq_tokens = [seq[self.token_size * idx:self.token_size * (idx + 1)] for idx in range(len(seq)//self.token_size)]
            X_transformed.append(seq_tokens)
        return X_transformed

    def fit(self, X, y=None):
        return self


class ProteinW2VRepresentation(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, model_path, agg_mode='sum'):
        self.model_path = model_path
        if agg_mode == 'sum':
            self.agg_fn = lambda x: np.sum(x, axis=0)
        else:
            self.agg_fn = lambda x: np.mean(x, axis=0)
        self.model = gensim.models.word2vec.Word2Vec.load(model_path)
        pass

    def transform(self, X, y=None):
        X_transformed = []
        for tokens_list in X:
            w2v = np.array([self.model.wv[str(token)] for token in tokens_list if token in self.model.wv])
            w2v = self.agg_fn(w2v)
            X_transformed.append(w2v)
        return np.array(X_transformed)

    def fit(self, X, y=None):
        return self
