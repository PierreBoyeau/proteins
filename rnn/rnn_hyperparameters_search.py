import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

import records_maker

RANDOM_STATE = 42
MAXLEN = 500
LR = 1e-2

DATA_PATH = '/home/pierre/riken/data/swiss/swiss_with_clans.tsv'
KEY_TO_PREDICT = 'clan'
log_dir = './logs_rnn_v2_swisstrain_with_weights'
transfer_path = None
GROUPS = None
SPLITTER = ShuffleSplit

# DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
# KEY_TO_PREDICT = 'is_allergenic'
# log_dir = './logs_transfer_group_shuffle'
# transfer_path = './logs_swisstrain_with_weights/weights.37-1.50.hdf5'
# GROUPS = 'DO'

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chars_to_idx = {char: idx+1 for (idx, char) in enumerate(chars)}
n_chars = len(chars)


def rnn_model(n_classes, n_features_wo_token):

    aa_ind = Input(shape=(MAXLEN,), name='aa_indice')
    embed = Embedding(len(matrix_embeddings), output_dim=n_chars+1, weights=[matrix_embeddings],
                      trainable=False, dtype='float32')(aa_ind)
    features = Input(shape=(MAXLEN, n_features_wo_token), dtype='float32')
    h = Concatenate()([embed, features])
    h = Conv1D(100, kernel_size=3, activation='relu')(h)
    h = LSTM(128, return_sequences=True)(h)
    h = Dropout(rate=0.5)(h)
    h = LSTM(128, return_sequences=False)(h)
    h = Dense(n_classes, activation='softmax')(h)

    mdl = Model(inputs=[aa_ind, features], outputs=h)
    optimizer = Adam(lr=LR)
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return mdl


def rnn_model_v2(n_classes, n_features_wo_token):
    # 1, 3, 5, 9, 15 and 21
    aa_ind = Input(shape=(MAXLEN,), name='aa_indice')
    embed = Embedding(len(matrix_embeddings), output_dim=n_chars+1, weights=[matrix_embeddings],
                      trainable=False, dtype='float32')(aa_ind)
    features = Input(shape=(MAXLEN, n_features_wo_token), dtype='float32')
    h = Concatenate()([embed, features])

    conv_layers = []
    for kernel_size in [1, 3, 5]:
        conv = Conv1D(50, kernel_size=kernel_size, activation='relu', padding='same')(h)
        conv_layers.append(conv)
    conv_layers.append(Conv1D(25, kernel_size=7, activation='relu', padding='same')(h))
    conv_layers.append(Conv1D(10, kernel_size=9, activation='relu', padding='same')(h))

    h = Concatenate()(conv_layers)
    h = Dropout(rate=0.3)(h)

    # h = Bidirectional(LSTM(128, return_sequences=True))(h)
    h = Bidirectional(LSTM(128, return_sequences=False))(h)
    h = Dense(n_classes, activation='softmax')(h)

    mdl = Model(inputs=[aa_ind, features], outputs=h)
    optimizer = Adam(lr=LR)
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return mdl


def transfer_model(n_classes_new, mdl_path, prev_model_output_layer='lstm_2'):
    prev_mdl = load_model(mdl_path)
    prev_mdl.layers.pop()

    top_mdl = Sequential()
    top_mdl.add(Dense(n_classes_new, activation='softmax'))

    mdl = Model(inputs=prev_mdl.input, outputs=top_mdl(prev_mdl.get_layer(prev_model_output_layer).output))

    optimizer = Adam(lr=LR)
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return mdl


def safe_char_to_idx(char):
    if char in chars_to_idx:
        return chars_to_idx[char]
    else:
        return 0


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t').dropna()
    # df = df.loc[df.seq_len >= 50, :]

    df.loc[:, 'sequences'] = df.sequences
    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[safe_char_to_idx(char) for char in sequence] for sequence in sequences], maxlen=MAXLEN)
    if GROUPS is None:
        groups = None
    else:
        groups = df.species.values

    features = np.array([records_maker.get_feat(tokens) for tokens in X])
    train_inds, test_inds = next(SPLITTER(random_state=RANDOM_STATE, test_size=0.2).split(sequences, y, groups))
    print(train_inds.shape, test_inds.shape)

    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    features_train, features_test = features[train_inds], features[test_inds]

    matrix_embeddings = np.zeros((n_chars+1, n_chars+1))
    matrix_embeddings[1:, 1:] = np.eye(n_chars, n_chars)

    if transfer_path is None:
        model = rnn_model_v2(n_classes=y.shape[1], n_features_wo_token=features.shape[2])
    else:
        model = transfer_model(mdl_path=transfer_path, n_classes_new=y.shape[1])
    model.summary()

    tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    ckpt = ModelCheckpoint(filepath=os.path.join(log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                           verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit([Xtrain, features_train], ytrain, batch_size=128, epochs=50,
              validation_data=([Xtest, features_test], ytest), callbacks=[tb, ckpt])
