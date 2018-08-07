import gensim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, roc_auc_score


"""
Useful functions for Word2Vec-based featurization
"""


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
            seq_tokens = [seq[self.token_size * idx:self.token_size * (idx + 1)]
                          for idx in range(len(seq)//self.token_size)]
            X_transformed.append(seq_tokens)
        return X_transformed

    def fit(self, X, y=None):
        return self


class ProteinW2VRepresentation(BaseEstimator, TransformerMixin):
    """
    Takes as input tokens (ie pseudo words) e.g. [['AAA', 'TTT', 'CGT'], ...]
    and returns sentence vector representations
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
            w2v = np.array([self.model.wv[str(token)] for token in tokens_list
                            if token in self.model.wv])
            w2v = self.agg_fn(w2v)
            X_transformed.append(w2v)
        return np.array(X_transformed)

    def fit(self, X, y=None):
        return self


def roc_score(y_test, y_score, **ax_kws):
    """
    Plots roc auc curve
    :param y_test:
    :param y_score:
    :param ax_kws:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc_value = roc_auc_score(y_test, y_score)
    fig, ax = plt.subplots(**ax_kws)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_value)

    fig.suptitle('ROC Curve')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize='medium')
    return fig, ax
