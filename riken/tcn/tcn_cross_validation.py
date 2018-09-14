from riken.tcn.tcn_keras import *
from riken.protein_io.data_op import pseudo_cv_groups
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from keras.callbacks import EarlyStopping

# PARAMS = {
#     "activation": ["selu", "relu", "tanh"],
#     "batch_size": 103,
#     "depth": 8,
#     "dropout_rate": 0.2,
#     "kernel_initializer": "glorot_uniform",
#     "kernel_size": 7,
#     "maxlen": 500,
#     "n_filters": 21,
#     "nb_epochs": 21,
#     "optim": Adam(1e-2),
#     "trainable_embeddings": True,
# }

PARAMS = {
    "activation": ["selu", "relu", "tanh"],
    "batch_size": np.arange(50, 150).tolist(),
    "depth": [7, 8, 9],
    "dropout_rate": np.linspace(0.1, 0.4, num=10).tolist(),
    "kernel_initializer": ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform'],
    "kernel_size": [3, 5, 7, 9],
    "maxlen": [1000],
    "n_filters": np.arange(10, 50).tolist(),
    "optim": [RMSprop(1e-2),  Adam(1e-2)],
    "trainable_embeddings": [True, False],
}

DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
KEY_TO_PREDICT = 'is_allergenic'
GROUPS = 'genre'
PSSM_FORMAT_FILE = '/home/pierre/riken/data/psiblast/riken_data_v2/{}_pssm.txt'
INDEX_COL = 0
# NB_EPOCHS = PARAMS.pop('nb_epochs')
# BATCH_SIZE = PARAMS.pop('batch_size')


def find_best_model(xtv, pssm_tv, ytv, groups_tv, grid):
    t_inds, v_inds = data_op.group_shuffle_indices(xtv, ytv, groups_tv, test_size=0.2)
    x_t = xtv[t_inds]
    pssm_t = pssm_tv[t_inds]
    y_t = ytv[t_inds]
    x_v = xtv[v_inds]
    pssm_v = pssm_tv[v_inds]
    y_v = ytv[v_inds]

    results = pd.DataFrame()
    for param in ParameterSampler(grid, n_iter=50):
        batch_size = param.pop('batch_size')
        callback = EarlyStopping(patience=5)
        mdl = tcn_model(n_classes=y.shape[1], **param)
        mdl.fit([x_t, pssm_t], y_t, batch_size=batch_size,
                callbacks=[callback],
                epochs=25, validation_data=[[x_v, pssm_v], y_v])
        score = roc_auc_score(y_v[:, 1], mdl.predict([x_v, pssm_v])[:, 1])

        param['batch_size'] = batch_size
        param['nb_epochs'] = callback.stopped_epoch
        results = results.append({"score": score, "params": param}, ignore_index=True)
    best_params = results.sort_values(by='score', ascending=False).iloc[0]["params"]
    return best_params


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t', index_col=INDEX_COL).dropna()
    df = df.loc[df.seq_len >= 50, :]

    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[prot_features.safe_char_to_idx(char) for char in sequence]
                       for sequence in sequences], maxlen=PARAMS['maxlen'][0])
    groups = df[GROUPS].values
    indices = df.index.values

    X, pssm, y = get_all_features(X, y, indices, pssm_format_fi=PSSM_FORMAT_FILE,
                                  maxlen=PARAMS['maxlen'][0])
    splits = pseudo_cv_groups(X, y, groups)

    info = []
    idx = 0
    for train_inds, test_inds in splits:
        perfs = dict()
        Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
        pssm_train, pssm_test = pssm[train_inds], pssm[test_inds]

        best_params = find_best_model(Xtrain, pssm_train, ytrain, groups[train_inds], PARAMS)
        perfs['params'] = best_params
        nb_epochs = best_params.pop('nb_epochs')
        batch_size = best_params.pop('batch_size')

        model = tcn_model(n_classes=y.shape[1], **best_params)
        history = model.fit([Xtrain, pssm_train], ytrain,
                            batch_size=batch_size,
                            epochs=nb_epochs,
                            validation_data=([Xtest, pssm_test], ytest))
        ypred = model.predict([Xtest, pssm_test])
        perfs['history'] = history.history
        perfs['loss'] = history.history['loss'][-1]
        perfs['val_loss'] = history.history['val_loss'][-1]
        try:
            perfs['roc_auc'] = roc_auc_score(ytest[:, 1], ypred[:, 1])
        except ValueError:
            perfs['roc_auc'] = None
        info.append(perfs)
        idx += 1
        pd.DataFrame(info).to_csv('best_model_NEW_cross_validation.csv', sep='\t')
    pd.DataFrame(info).to_csv('best_model_NEW_cross_validation.csv', sep='\t')