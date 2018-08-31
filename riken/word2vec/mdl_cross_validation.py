import catboost
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from riken.protein_io.data_op import pseudo_cv_groups
from riken.word2vec import classification_tools


def overlapping_tokenizer(seq, k=3):
    """
    'ABCDE' ==> ['ABC', 'BCE', 'CDE']
    """
    return [seq[iind:iind + k] for iind in range(len(seq) - k + 1)]


def tokenizer4(seq):
    return overlapping_tokenizer(seq, k=4)


def get_clf(mode_name):
    if mode_name == 'svm':
        model = Pipeline([
            ('TFIDF', TfidfVectorizer(max_features=1000000, lowercase=False)),
            # ('SVM', LinearSVC(tol=1e-6, max_iter=50000)),
            ('SVM', LinearSVC(
                # tol=1e-6, max_iter=-1, random_state=42,
            ))
        ])
        model = model.set_params(**SVM_PARAMS)
        return model
    else:
        return catboost.CatBoostClassifier(**BST_PARAMS)


def preprocess_data(x, mode_name):
    if mode_name == 'svm':
        return x
    else:
        feature_tool = Pipeline([
            ('ProteinTokenizer', classification_tools.ProteinTokenizer(token_size=4)),
            ('ProteinVectorization', classification_tools.ProteinW2VRepresentation(model_path=MODEL_PATH,
                                                                                   agg_mode=AGG_MODE)),
        ])
        return feature_tool.transform(x)


DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
INDEX_COL = 0
KEY_TO_PREDICT = 'is_allergenic'
GROUPS = 'genre'
AGG_MODE = 'sum'
MODEL_PATH = '/home/pierre/riken/riken/word2vec/prot_vec_model_10_epochs_l_4.model'
BST_PARAMS = {'l2_leaf_reg': 15.0,
              'class_weights': [1.0, 5.0],
              'random_strength': 10,
              'depth': 5,
              'iterations': 1000,
              'bagging_temperature': 1.0}
SVM_PARAMS = {
              'SVM__C': 1.0,
              # 'SVM__C': 15.0,
              'SVM__class_weight': 'balanced',
              'SVM__dual': False,
              'SVM__loss': 'squared_hinge', 'SVM__penalty': 'l1', 'TFIDF__ngram_range': (1, 4),
              'TFIDF__tokenizer': tokenizer4, 'TFIDF__use_idf': True}
MODE = 'svm'

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t', index_col=INDEX_COL).dropna()
    df = df.loc[df.seq_len >= 50, :]
    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT].values
    X = sequences
    groups = df[GROUPS].values
    splits = pseudo_cv_groups(X, y, groups)

    X_preprocessed = preprocess_data(X, MODE)
    info = []
    idx = 0
    for train_inds, test_inds in splits:
        perfs = dict()
        Xtrain, Xtest, ytrain, ytest = X_preprocessed[train_inds], X_preprocessed[test_inds], y[train_inds], y[test_inds]
        clf = get_clf(MODE)
        clf.fit(Xtrain, ytrain)
        # ypred = clf.predict_proba(Xtest)[:, 1]
        ypred = clf.decision_function(Xtest)

        try:
            print(ypred[:3])
            perfs['roc_auc'] = roc_auc_score(ytest, ypred)
        except ValueError:
            perfs['roc_auc'] = None
        info.append(perfs)
        idx += 1
    pd.DataFrame(info).to_csv('best_model_allerdictor_cross_validation.csv', sep='\t')
