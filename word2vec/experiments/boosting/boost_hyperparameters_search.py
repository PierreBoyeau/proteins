import catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from word2vec import classification_tools

###############

RANDOM_STATE = 42
model_path = '/home/pierre/riken/word2vec/prot_vec_model_10_epochs_l_4.model'
data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
BEST_MODEL_PROPERTIES_PATH = './best_model_boosting_group_kfold_w2v_10_epochs_l_4.txt'
CV_OUTPUT_PATH = './models_boosting_group_kfold_w2v_10_epochs_l_4.csv'
agg_mode = 'sum'
K_FOLD = 5
KFOLD = GroupKFold(n_splits=K_FOLD)


###############

df = pd.read_csv(data_path, sep='\t').dropna()
df.loc[:, 'seq_len'] = df.sequences.apply(len)
df = df.loc[df.seq_len >= 50, :]
X, y = df['sequences'].values, df['is_allergenic'].values
groups = df.species.values

feature_tool = Pipeline([
    ('ProteinTokenizer', classification_tools.ProteinTokenizer(token_size=4)),
    ('ProteinVectorization', classification_tools.ProteinW2VRepresentation(model_path=model_path,
                                                                           agg_mode=agg_mode)),
])
X_w2v = feature_tool.transform(X)
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_w2v, y, test_size=0.3, random_state=RANDOM_STATE)
train_inds, test_inds = next(GroupShuffleSplit(random_state=RANDOM_STATE).split(X_w2v, y, groups))
Xtrain, Xtest, ytrain, ytest = X_w2v[train_inds], X_w2v[test_inds], y[train_inds], y[test_inds]


clf = catboost.CatBoostClassifier(verbose=False)

grid = {
    'iterations': [1000],
    'class_weights': [[1.0, 5.0]],
    'depth': np.arange(4, 6).tolist(),
    'l2_leaf_reg': [1e-1, 1e0, 1e1],
    'random_strength': [0, 1, 10],
    'bagging_temperature': [0.0, 0.5, 1.0],
}

cv_splits = KFOLD.split(Xtrain, ytrain, groups[train_inds])
cv = GridSearchCV(clf,
                  grid,
                  scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                  refit='roc_auc',
                  cv=cv_splits,  # StratifiedKFold(n_splits=10, random_state=RANDOM_STATE),
                  verbose=3)
cv.fit(Xtrain, ytrain)

print(cv.best_params_, cv.best_score_)
pd.DataFrame(cv.cv_results_).to_csv(CV_OUTPUT_PATH, sep='\t', index=False)

clf = cv.best_estimator_
clf.fit(Xtrain, ytrain)
file_content = '\nBest params: {}\nROC_AUC_SCORE : {}\n{}'\
        .format(cv.best_params_,
                roc_auc_score(ytest, clf.predict_proba(Xtest)[:, 1]),
                classification_report(ytest, clf.predict(Xtest)))
with open(BEST_MODEL_PROPERTIES_PATH, "w") as text_file:
    text_file.write(file_content)
