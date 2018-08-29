from sklearn.pipeline import Pipeline
from riken.word2vec import classification_tools
from sklearn.metrics import roc_auc_score
import pandas as pd
from riken.protein_io.data_op import pseudo_cv_groups
import catboost

DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
INDEX_COL = 0
KEY_TO_PREDICT = 'is_allergenic'
GROUPS = 'genre'
AGG_MODE = 'sum'
MODEL_PATH = '/home/pierre/riken/riken/word2vec/prot_vec_model_10_epochs_l_4.model'
MDL_PARAMS = {'l2_leaf_reg': 1.0,
              'class_weights': [1.0, 5.0],
              'random_strength': 10,
              'depth': 5,
              'iterations': 1000,
              'bagging_temperature': 1.0}

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t', index_col=INDEX_COL).dropna()
    df = df.loc[df.seq_len >= 50, :]
    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT].values
    X = sequences
    groups = df[GROUPS].values
    splits = pseudo_cv_groups(X, y, groups)

    feature_tool = Pipeline([
        ('ProteinTokenizer', classification_tools.ProteinTokenizer(token_size=4)),
        ('ProteinVectorization', classification_tools.ProteinW2VRepresentation(model_path=MODEL_PATH,
                                                                               agg_mode=AGG_MODE)),
    ])
    X_w2v = feature_tool.transform(X)

    info = []
    idx = 0
    for train_inds, test_inds in splits:
        perfs = dict()
        Xtrain, Xtest, ytrain, ytest = X_w2v[train_inds], X_w2v[test_inds], y[train_inds], y[test_inds]
        clf = catboost.CatBoostClassifier(**MDL_PARAMS)
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict_proba(Xtest)[:, 1]
        try:
            perfs['roc_auc'] = roc_auc_score(ytest, ypred)
        except ValueError:
            perfs['roc_auc'] = None
        info.append(perfs)
        idx += 1
    pd.DataFrame(info).to_csv('best_model_protvec_cross_validation.csv', sep='\t')
