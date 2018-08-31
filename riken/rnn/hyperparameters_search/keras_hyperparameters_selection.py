from riken.rnn.rnn_keras_with_psiblast import *

from keras.optimizers import SGD, RMSprop
from keras.callbacks import  EarlyStopping
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score
import time


def generate_configs():
    depth = np.random.randint(1, 4)
    if depth == 1:
        n_filters = np.random.randint(10, 101)
        kernel_size = np.random.choice([1, 3, 5, 7, 9])
    else:
        n_filters = np.random.randint(10, 101, depth).tolist()
        kernel_size = np.random.choice([1, 3, 5, 7, 9], depth).tolist()
    return {"n_filters": n_filters, "kernel_size": kernel_size}


if __name__ == '__main__':
    RESULTS_PATH = 'results.csv'
    RANDOM_STATE = 42
    MAXLEN = 500
    DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
    KEY_TO_PREDICT = 'is_allergenic'
    SPLITTER = data_op.group_shuffle_indices
    PSSM_FORMAT_FILE = '/home/pierre/riken/data/psiblast/riken_data_v2/{}_pssm.txt'
    INDEX_COL = 0

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    K.set_session(tf.Session(config=config))

    df = pd.read_csv(DATA_PATH, sep='\t', index_col=INDEX_COL).dropna()
    df = df.loc[df.seq_len >= 50, :]

    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[prot_features.safe_char_to_idx(char) for char in sequence]
                       for sequence in sequences], maxlen=MAXLEN)
    indices = df.index.values
    groups = df['species'].values

    train_inds, test_inds = np.where(df.is_train)[0], np.where(df.is_train == False)[0]

    print('{} train examples and {} test examples'.format(len(train_inds), len(test_inds)))
    assert len(np.intersect1d(train_inds, test_inds)) == 0
    print(train_inds.shape, test_inds.shape)

    X, pssm, y = get_all_features(X, y, indices, pssm_format_fi=PSSM_FORMAT_FILE)

    print('NORMALIZATION ...')
    X = (X - X[train_inds].mean(axis=0)) / X[train_inds].std(axis=0)
    pssm = (pssm - pssm[train_inds].mean(axis=0)) / pssm[train_inds].std(axis=0)

    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    pssm_train, pssm_test = pssm[train_inds], pssm[test_inds]

    trainval_inds, val_inds = SPLITTER(Xtrain, ytrain, groups[train_inds], test_size=0.35)
    Xtrainval, Xval = Xtrain[trainval_inds], Xtrain[val_inds]
    pssm_trainval, pssm_val = pssm_train[trainval_inds], pssm_train[val_inds]
    ytrainval, yval = ytrain[trainval_inds], ytrain[val_inds]

    print('{} trainval examples and {} val examples'.format(len(trainval_inds), len(val_inds)))

    grid_params = {
        'n_classes': [2],
        # 'n_filters': np.arange(25, 100),
        # 'kernel_size': [3, 4, 5, 6],
        'conv_params': [generate_configs() for _ in range(3000)],
        'activation': ['relu', 'selu', 'tanh'],
        'n_cells': np.arange(8, 32),
        'trainable_embeddings': [True, False],
        'dropout_rate': np.linspace(0.1, 0.5, 10),
        'conv_kernel_initializer': ['glorot_uniform', 'glorot_normal'],
        'lstm_kernel_initializer': ['glorot_uniform', 'glorot_normal'],

        'batch_size': np.arange(32, 100),
        'optim': [Adam, RMSprop, SGD],
        'lr': [1e-1, 1e-2, 1e-3],    }

    grid = ParameterSampler(grid_params, n_iter=10000)
    res_df = pd.DataFrame()
    for param in grid:
        try:
            print(param)

            lr = param.pop('lr')
            optim_fn = param.pop('optim')
            param['optim'] = optim_fn(lr)

            conv_params = param.pop('conv_params')
            param['n_filters'] = conv_params['n_filters']
            param['kernel_size'] = conv_params['kernel_size']

            batch_size = param.pop('batch_size')
            mdl = rnn_model_attention_psiblast(**param)
            callback = EarlyStopping(patience=5)

            history = mdl.fit([Xtrainval, pssm_trainval], ytrainval, batch_size=batch_size,
                              callbacks=[callback],
                              epochs=25, validation_data=[[Xval, pssm_val], yval])

            param['lr'] = lr
            param['optim'] = optim_fn
            param['nb_epochs'] = callback.stopped_epoch
            param['test_score'] = roc_auc_score(yval[:, 1], mdl.predict([Xval, pssm_val])[:, 1])
            param['batch_size'] = batch_size
            param['history'] = history.history
            res_df = res_df.append(param, ignore_index=True)
            res_df.to_csv(RESULTS_PATH, sep='\t')
        except Exception as e:
            print('ERROR!!!!!')
            print(e)
            time.sleep(10)
            pass

