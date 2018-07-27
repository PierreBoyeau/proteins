import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from riken.word2vec import classification_tools

###############

RANDOM_STATE = 42

model_path = '/home/pierre/riken/word2vec/prot_vec_model.model'
data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
BEST_MODEL_PROPERTIES_PATH = './best_model_svm.txt'
CV_OUTPUT_PATH = './tree_depth_influence.csv'

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


model = LinearSVC(tol=1e-6, max_iter=50000, class_weight='balanced')
grid = [
{
    'loss': ['hinge'],
    'dual': [True],
    'penalty': ['l2'],
    'C': np.geomspace(start=1e-3, stop=5e1, num=10)
        },
        {
    'loss': ['squared_hinge'],
    'dual': [False],
    'penalty': ['l1'],
    'C': np.geomspace(start=1e-3, stop=5e1, num=10)
        },
    ]

cv = GridSearchCV(model,
                  grid,
                  scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                  refit='roc_auc',
                  cv=StratifiedKFold(n_splits=10),
                  verbose=3, n_jobs=8)
cv.fit(Xtrain, ytrain)
print(cv.best_params_, cv.best_score_)
pd.DataFrame(cv.cv_results_).to_csv(CV_OUTPUT_PATH, sep='\t', index=False)
model.fit(Xtrain, ytrain)

file_content = 'Best params: {}\n{}'.format(cv.best_params_,
                                            classification_report(ytest, model.predict(Xtest)))
with open(BEST_MODEL_PROPERTIES_PATH, "w") as text_file:
    text_file.write(file_content)
