# IMPORTS AND PARAMETERS

import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from riken.protein_io import data_op

K = 5
RANDOM_STATE = 42
BEST_MODEL_PROPERTIES_PATH = './best_model_rbf_svm_group_kfold_n_grams.txt'
CV_OUTPUT_PATH = './rbf_parameters_group_kfold_n_grams.csv'
data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
KFOLD = GroupKFold(n_splits=5)
# KFOLD = StratifiedKFold(n_splits=10, random_state=RANDOM_STATE)


def overlapping_tokenizer(seq, k=K):
    """
    'ABCDE' ==> ['ABC', 'BCE', 'CDE']
    """
    return [seq[idx:idx + K] for idx in range(len(seq) - k + 1)]


def tokenizer6(seq):
    return overlapping_tokenizer(seq, k=6)


def tokenizer5(seq):
    return overlapping_tokenizer(seq, k=5)


def tokenizer4(seq):
    return overlapping_tokenizer(seq, k=4)


def tokenizer3(seq):
    return overlapping_tokenizer(seq, k=3)


if __name__ == '__main__':
    # DATA IMPORT AND PREPROCESSING
    df = pd.read_csv(data_path, sep='\t').dropna()
    df.loc[:, 'seq_len'] = df.sequences.apply(len)
    df = df.loc[df.seq_len >= 50, :]
    sequences, y = df['sequences'].values, df['is_allergenic'].values
    groups = df.species.values

    # PIPELINE DESCRIPTION
    pipe = Pipeline([
        ('TFIDF', TfidfVectorizer(max_features=32000, lowercase=False)),
        # ('SVM', LinearSVC(tol=1e-6, max_iter=50000)),
        ('RBF_SVM', SVC(tol=1e-6, max_iter=-1, cache_size=5000, random_state=RANDOM_STATE, kernel='rbf'))
    ])

    grid = {
        'TFIDF__tokenizer': [
                      # tokenizer6,
                      # tokenizer5,
                      tokenizer4,
                      # tokenizer3
        ],
        'TFIDF__use_idf': [
                           # False,
                           True
                           ],
        'TFIDF__ngram_range': [(1, 4)],  #[(1,1), (1,2), (1,3), (1,4)],
        # 'SVM__loss': ['squared_hinge'],
        # 'SVM__dual': [False],
        # 'SVM__penalty': ['l1'],
        # 'SVM__class_weight': ['balanced'],  # [None, 'balanced'],
        # 'SVM__C': [1e-1, 1e0, 5.0, 10.0, 15.0],

        'RBF_SVM__C': [1e-1, 1.0, 10., 100.],
        'RBF_SVM__gamma': ['auto', 1e-2, 1e-1, 1]
    }

    train_inds, test_inds = data_op.group_shuffle_indices(sequences, y, groups)
    Xtrain, Xtest, ytrain, ytest = sequences[train_inds], sequences[test_inds], y[train_inds], y[test_inds]

    cv_splits = KFOLD.split(Xtrain, ytrain, groups[train_inds])

    # id_bf = None
    # for train_id, test_id in cv_splits:
    #     print('\n', np.all(np.unique(train_id)==id_bf))
    #     print(train_id.shape, test_id.shape)
    #     id_bf = np.unique(train_id)

    cv = GridSearchCV(pipe,
                      grid,
                      scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                      refit='roc_auc',
                      cv=cv_splits,  # StratifiedKFold(n_splits=10, random_state=RANDOM_STATE),
                      verbose=3, n_jobs=16, return_train_score=True)
    cv.fit(Xtrain, ytrain)
    print(cv.best_params_, cv.best_score_)
    pd.DataFrame(cv.cv_results_).to_csv(CV_OUTPUT_PATH, sep='\t', index=False)
    pipe = cv.best_estimator_
    pipe.fit(Xtrain, ytrain)

    file_content = '\nBest params: {}\nROC_AUC_SCORE : {}\n{}'\
        .format(cv.best_params_,
                roc_auc_score(ytest, pipe.decision_function(Xtest)),
                classification_report(ytest, pipe.predict(Xtest)))
    with open(BEST_MODEL_PROPERTIES_PATH, "w") as text_file:
        text_file.write(file_content)
