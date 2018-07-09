import catboost
import sys
import glob
sys.path.append('/home/pierre/riken/word2vec')
sys.path.append('/home/pierre/riken/io')

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import classification_tools
import reader

import matplotlib.pyplot as plt


###############

RANDOM_STATE = 42

model_path = '/home/pierre/riken/word2vec/prot_vec_model.model'
data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
BEST_MODEL_PROPERTIES_PATH = './best_model_boosting.txt'
CV_OUTPUT_PATH = './models_boosting.csv'

K_FOLD = 10
agg_mode = 'sum'

###############

df = pd.read_csv(data_path, sep='\t')
df.loc[:, 'seq_len'] = df.sequences.apply(len)
df = df.loc[df.seq_len >= 50, :]

X, y = df['sequences'].values, df['is_allergenic'].values
feature_tool = Pipeline([
    ('ProteinTokenizer', classification_tools.ProteinTokenizer(token_size=3)),
    ('ProteinVectorization', classification_tools.ProteinW2VRepresentation(model_path=model_path,
                                                                           agg_mode=agg_mode)),
])
X_w2v = feature_tool.transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_w2v, y, test_size=0.3, random_state=RANDOM_STATE)


clf = catboost.CatBoostClassifier(iterations=2000, verbose=False)

grid = {
    'depth': np.arange(4, 11).tolist(),
    'l2_leaf_reg': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e1],
}

cv = GridSearchCV(clf,
                  grid,
                  scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                  refit='roc_auc',
                  cv=StratifiedKFold(n_splits=10, random_state=RANDOM_STATE),
                  verbose=3)
cv.fit(Xtrain, ytrain)

clf.fit(Xtrain, ytrain, eval_set=(Xtest, ytest))


print(cv.best_params_, cv.best_score_)
pd.DataFrame(cv.cv_results_).to_csv(CV_OUTPUT_PATH, sep='\t', index=False)
clf.fit(Xtrain, ytrain)

file_content = 'Best params: {}\n{}'.format(cv.best_params_,
                                            classification_report(ytest, clf.predict(Xtest)))
with open(BEST_MODEL_PROPERTIES_PATH, "w") as text_file:
    text_file.write(file_content)
