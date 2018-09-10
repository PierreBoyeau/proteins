from riken.rnn.rnn_keras_with_psiblast import *
from riken.protein_io.data_op import pseudo_cv_groups
from sklearn.metrics import roc_auc_score


## BEST MODEL 2
PARAMS = {
    'activation': 'tanh',
    'conv_kernel_initializer': 'glorot_uniform',
    'dropout_rate': 0.3222222222,
    'kernel_size': 6,
    'lstm_kernel_initializer': 'glorot_normal',
    'n_cells': 25,
    'n_filters': 79,
    'nb_epochs': 13,
    'optim': RMSprop(),
    'trainable_embeddings': False,
    'batch_size': 85,

    'maxlen': 500
}

## BEST PARAMS 2BIS
# PARAMS = {
#     'activation': "selu",
#     'batch_size': 93,
#     'conv_kernel_initializer': "glorot_normal",
#     'dropout_rate': 0.1,
#     'kernel_size': 5,
#     'lstm_kernel_initializer': "glorot_uniform",
#     'n_cells': 17,
#     # 'n_classes': 2,
#     'n_filters': 33,
#     'nb_epochs': 12,
#     'optim': Adam(1e-2),
#     'trainable_embeddings': True,
# }

LR = 1e-3
DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
KEY_TO_PREDICT = 'is_allergenic'
GROUPS = 'genre'
PSSM_FORMAT_FILE = '/home/pierre/riken/data/psiblast/riken_data_v2/{}_pssm.txt'
INDEX_COL = 0
NB_EPOCHS = PARAMS.pop('nb_epochs')
BATCH_SIZE = PARAMS.pop('batch_size')

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t', index_col=INDEX_COL).dropna()
    df = df.loc[df.seq_len >= 50, :]

    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[prot_features.safe_char_to_idx(char) for char in sequence]
                       for sequence in sequences], maxlen=PARAMS['maxlen'])
    groups = df[GROUPS].values
    indices = df.index.values

    X, pssm, y = get_all_features(X, y, indices, pssm_format_fi=PSSM_FORMAT_FILE,
                                  maxlen=PARAMS['maxlen'])
    splits = pseudo_cv_groups(X, y, groups)

    info = []
    idx = 0
    for train_inds, test_inds in splits:
        perfs = dict()
        Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
        pssm_train, pssm_test = pssm[train_inds], pssm[test_inds]
        model = rnn_model_attention_psiblast(n_classes=y.shape[1], **PARAMS)
        history = model.fit([Xtrain, pssm_train], ytrain,
                            batch_size=BATCH_SIZE,
                            epochs=NB_EPOCHS,
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
    pd.DataFrame(info).to_csv('best_model_v2BIS_cross_validation.csv', sep='\t')
