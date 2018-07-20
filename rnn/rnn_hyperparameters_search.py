import os
import sys
sys.path.append('/home/pierre/riken/io')
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import Concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.metrics
from keras.callbacks import TensorBoard, ModelCheckpoint

import records_maker

RANDOM_STATE = 42
MAXLEN = 500

# DATA_PATH = '/home/pierre/riken/data/swiss/swiss_with_clans.tsv'
# KEY_TO_PREDICT = 'clan'

DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
KEY_TO_PREDICT = 'is_allergenic'


chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
log_dir = './logs7'
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

    model = Model(inputs=[aa_ind, features], outputs=h)

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def safe_char_to_idx(char):
    if char in chars_to_idx:
        return chars_to_idx[char]
    else:
        return 0


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep='\t')
    # df.loc[:, 'sequences'] = df.sequences_x
    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[safe_char_to_idx(char) for char in sequence] for sequence in sequences], maxlen=MAXLEN)
    features = np.array([records_maker.get_feat(tokens) for tokens in X])

    train_inds, test_inds = next(ShuffleSplit(random_state=RANDOM_STATE).split(X, y))
    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    features_train, features_test = features[train_inds], features[test_inds]

    matrix_embeddings = np.zeros((n_chars+1, n_chars+1))
    matrix_embeddings[1:, 1:] = np.eye(n_chars, n_chars)

    model = rnn_model(n_classes=y.shape[1], n_features_wo_token=features.shape[2])
    model.summary()

    tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    ckpt = ModelCheckpoint(filepath=os.path.join(log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'))
    model.fit([Xtrain, features_train], ytrain, batch_size=128, epochs=50,
              validation_data=([Xtest, features_test], ytest), callbacks=[tb])
