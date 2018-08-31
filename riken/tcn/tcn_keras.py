import argparse
import os

import keras.backend as K
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation, Add, BatchNormalization, Conv1D, Dense, Dropout, Input, Lambda, \
    RepeatVector, Permute, Multiply, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score

from riken.protein_io import data_op
from riken.protein_io import prot_features
from riken.rnn.rnn_keras_with_psiblast import get_embeddings, transfer_model, get_all_features

"""
"""
RANDOM_STATE = 42
PARAMS = {
    'depth': 8,
    'n_filters': 25,
    'kernel_size': 7,
    'dropout_rate': 0.5,
    'optim': Adam(),
    'nb_epochs': 100,
    'batch_size': 64,
}


def residual_block(inp, dilatation, kernel_size, n_filters, dropout_rate, do1conv=True):
    conv = inp
    for _ in range(2):
        conv = Conv1D(n_filters, kernel_size=kernel_size, dilation_rate=dilatation, padding='causal')(conv)
        # here do weight norm (later)
        # instead here use of batch norm because already implemented
        conv = BatchNormalization()(conv)
        conv = Activation(activation='relu')(conv)
        conv = Dropout(rate=dropout_rate)(conv)

    if do1conv:
        rescaled_input = Conv1D(n_filters, kernel_size=1)(inp)
    else:
        rescaled_input = inp

    last = Add()([conv, rescaled_input])
    return last


def tcn_model(n_classes, depth, n_filters, kernel_size, dropout_rate=0.0, optim=Adam(), 
              maxlen=500, trainable_embeddings=False):
    aa_ind = Input(shape=(maxlen,), name='aa_indice')
    h = get_embeddings(aa_ind, trainable_embeddings=trainable_embeddings)

    psiblast_prop = Input(shape=(maxlen, 42), name='psiblast_prop', dtype=np.float32)

    h = Concatenate()([h, psiblast_prop])
    for it in range(depth):
        # 1conv done only first layer (elsewhere number of filters stays the same
        do1conv = (it == 0)
        h = residual_block(h, dilatation=2**it, n_filters=n_filters, kernel_size=kernel_size,
                           dropout_rate=dropout_rate,
                           do1conv=do1conv)
    attention = Dense(1)(h)
    attention = Lambda(lambda x: K.squeeze(x, axis=2))(attention)
    attention = Activation(activation='softmax')(attention)
    attention = RepeatVector(n_filters)(attention)
    attention = Permute((2, 1))(attention)
    last = Multiply()([attention, h])
    last = Lambda(lambda x: K.sum(x, axis=1), output_shape=(n_filters,))(last)

    h = Dense(n_classes, activation='softmax')(last)
    mdl = Model(inputs=[aa_ind, psiblast_prop], outputs=h)
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optim,
                metrics=['accuracy'])

    return mdl


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_len', type=int, default=1000, help='max sequence lenght')
    parser.add_argument('-data_path', type=str, help='path to tsv data')
    parser.add_argument('-index_col', type=int, default=0,
                        help='path to ckpt if doing transfer learning')
    parser.add_argument('-key_to_predict', type=str, help='key to predict (y)')
    parser.add_argument('-log_dir', type=str, help='path to save ckpt and summaries')
    parser.add_argument('-transfer_path', type=str, default=None,
                        help='path to ckpt if doing transfer learning')
    parser.add_argument('-layer_name', type=str, default=None,
                        help='Name of layer to use for transfer')
    parser.add_argument('-groups', type=str, default='NO', help='should we use groups')
    parser.add_argument('-pssm_format_file', type=str, help='path format of pssm files')
    return parser.parse_args()


def main():
    args = parse_arguments()
    groups_mode = args.groups if args.groups != 'NO' else None
    splitter = data_op.shuffle_indices if groups_mode is None else data_op.group_shuffle_indices
    nb_epochs = PARAMS.pop('nb_epochs')
    batch_size = PARAMS.pop('batch_size')

    df = pd.read_csv(args.data_path, sep='\t', index_col=args.index_col).dropna()
    df = df.loc[df.seq_len >= 50, :]

    sequences, y = df['sequences'].values, df[args.key_to_predict]
    y = pd.get_dummies(y).values
    X = pad_sequences([[prot_features.safe_char_to_idx(char) for char in sequence]
                       for sequence in sequences], maxlen=args.max_len)
    indices = df.index.values
    if groups_mode == 'predefined':
        train_inds, test_inds = np.where(df.is_train)[0], np.where(df.is_train == False)[0]
    else:
        groups = None if groups_mode is None else df[groups_mode].values
        train_inds, test_inds = splitter(sequences, y, groups)
    print('{} train examples and {} test examples'.format(len(train_inds), len(test_inds)))
    assert len(np.intersect1d(train_inds, test_inds)) == 0
    X, pssm, y = get_all_features(X, y, indices, pssm_format_fi=args.pssm_format_file)
    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    pssm_train, pssm_test = pssm[train_inds], pssm[test_inds]

    if args.transfer_path is None:
        model = tcn_model(n_classes=y.shape[1], **PARAMS)
    else:
        model = transfer_model(n_classes_new=y.shape[1], mdl_path=args.transfer_path,
                               prev_model_output_layer='lambda_1')
    print(model.summary())

    tb = TensorBoard(log_dir=args.log_dir)
    ckpt = ModelCheckpoint(
        filepath=os.path.join(args.log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
        period=1)
    model.fit([Xtrain, pssm_train], ytrain,  # bf [Xtrain, features_train], ...
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=([Xtest, pssm_test], ytest),
              callbacks=[tb, ckpt])
    print(roc_auc_score(ytest[:, 1], model.predict(Xtest)[:, 1]))


if __name__ == '__main__':
    main()
