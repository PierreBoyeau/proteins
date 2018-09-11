import numpy as np
import keras.backend as K
import numpy as np
import pandas as pd
from keras.activations import softmax
from keras.layers import Bidirectional, Dense, CuDNNLSTM
from keras.layers import RepeatVector, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from riken.protein_io import prot_features
chars = prot_features.chars
from riken.rnn.rnn_keras_with_psiblast import rnn_model_attention_psiblast, get_all_features
from sklearn.preprocessing import OneHotEncoder


def autoencoder(encoder_params, decoder_params):

    encoder_mdl = rnn_model_attention_psiblast(**encoder_params)
    output_previous = encoder_mdl.get_layer('lambda_2').output
    print(output_previous)
    means = Dense(units=decoder_params['n_hidden'])(output_previous)
    # log_smg = Dense(decoder_params['n_hidden'])(output_previous)
    # epsilon = K.random_normal(shape=h_shape, mean=0.0, stddev=1.0)
    # h = means + (K.exp(log_smg) * epsilon) # TODO : lambda layer
    h = means

    h = RepeatVector(MAXLEN)(h)
    h = Bidirectional(CuDNNLSTM(decoder_params['n_cells'], return_sequences=True))(h)
    # h = Dense(len(chars), activation=softmax(h, axis=-1))(h)
    h = Dense(len(chars)+1, activation='softmax')(h)
    autoencoder_mdl = Model(inputs=encoder_mdl.inputs, outputs=h)
    autoencoder_mdl.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return autoencoder_mdl


def one_hot_2d(data):
    data_np = np.array(data)
    n_labels = len(np.unique(data_np))
    oh_enc = OneHotEncoder(n_values=n_labels, sparse=False)
    res = [oh_enc.fit_transform(row.reshape(-1, 1)) for row in data_np]
    return np.array(res)

ENC_PARAMS = {
    'n_classes': 2,
    'activation': 'tanh',
    'conv_kernel_initializer': 'glorot_uniform',
    'dropout_rate': 0.3222222222,
    'kernel_size': 6,
    'lstm_kernel_initializer': 'glorot_normal',
    'n_cells': 25,
    'n_filters': 79,
    'optim': Adam(),
    'trainable_embeddings': False,
}


DEC_PARAMS = {
    'n_hidden': 100,
    'n_cells': 25
}


if __name__ == '__main__':
    DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
    PSSM_FILE_FORMAT = '/home/pierre/riken/data/psiblast/riken_data_v2/{}_pssm.txt'
    MAXLEN = 500

    df = pd.read_csv(DATA_PATH, sep='\t', index_col=0).dropna()
    df = df.loc[df.seq_len >= 50, :]

    sequences, y = df['sequences'].values, df['is_allergenic'].values
    y = pd.get_dummies(y).values
    X = pad_sequences([[prot_features.safe_char_to_idx(char) for char in sequence]
                       for sequence in sequences], maxlen=MAXLEN)
    indices = df.index.values
    train_inds, test_inds = np.where(df.is_train)[0], np.where(df.is_train == False)[0]
    print('{} train examples and {} test examples'.format(len(train_inds), len(test_inds)))
    assert len(np.intersect1d(train_inds, test_inds)) == 0
    print(train_inds.shape, test_inds.shape)

    X, pssm, y = get_all_features(X, y, indices,
                                  pssm_format_fi=PSSM_FILE_FORMAT)
    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    pssm_train, pssm_test = pssm[train_inds], pssm[test_inds]

    print(pssm_train[0])
    print(pssm_test[0])

    model = autoencoder(encoder_params=ENC_PARAMS,
                        decoder_params=DEC_PARAMS)
    print(model.summary())

    Xtrain_onehot = one_hot_2d(Xtrain)
    model.fit([Xtrain, pssm_train], Xtrain_onehot,
              batch_size=64,
              epochs=100,
              # validation_data=([Xtest, pssm_test], Xtest)
              )
