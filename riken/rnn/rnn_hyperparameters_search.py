import os
import numpy as np
import pandas as pd

from tensorflow import flags
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import CuDNNLSTM
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import Permute, Reshape, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from riken.rnn import records_maker
from riken.protein_io import data_op

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

"""
python rnn_hyperparameters_search.py \
-max_len 500 \
-lr 0.001 \
-data_path /home/pierre/riken/data/pfam/pfam_data.tsv \
-key_to_predict clan \
-log_dir logs_rnn_v2_pfam


python rnn_hyperparameters_search.py \
-max_len 500 \
-lr 0.001 \
-data_path /home/pierre/riken/data/riken_data/complete_from_xlsx.tsv \
-key_to_predict is_allergenic \
-log_dir logs_v2_transfer_group_shuffle \
-groups species \
-layer_name bidirectional_1 \
-transfer_path 


python rnn_hyperparameters_search.py \
-max_len 500 \
-lr 0.001 \
-data_path /home/pierre/riken/data/riken_data/complete_from_xlsx.tsv \
-key_to_predict is_allergenic \
-log_dir logs_rnn_v2_transfer_attention \
-groups species \
-layer_name bidirectional_1 \
-transfer_path
"""

# DATA_PATH = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
# KEY_TO_PREDICT = 'is_allergenic'
# log_dir = './logs_transfer_group_shuffle'
# TRANSFER_PATH = './logs_swisstrain_with_weights/weights.37-1.50.hdf5'
# GROUPS = 'DO'

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chars_to_idx = {char: idx+1 for (idx, char) in enumerate(chars)}
n_chars = len(chars)

# STATIC_AA_TO_FEAT_M = records_maker.create_blosom_80_mat()
STATIC_AA_TO_FEAT_M = records_maker.create_overall_static_aa_mat(normalize=True)
# TODO: Ensure that new features (chemical properties of AA bring something to the model) via CV?

ONEHOT_M = np.zeros((n_chars + 1, n_chars + 1))
ONEHOT_M[1:, 1:] = np.eye(n_chars, n_chars)


def rnn_model(n_classes, n_features_wo_token=None, attention=False):
    aa_ind = Input(shape=(MAXLEN,), name='aa_indice')
    embed = Embedding(len(ONEHOT_M), output_dim=n_chars + 1, weights=[ONEHOT_M],
                      trainable=False, dtype='float32')(aa_ind)

    static_feat_from_aa = Embedding(STATIC_AA_TO_FEAT_M.shape[0], output_dim=STATIC_AA_TO_FEAT_M.shape[1],
                                    weights=[STATIC_AA_TO_FEAT_M],
                                    trainable=False, dtype='float32')(aa_ind)

    # features = Input(shape=(MAXLEN, n_features_wo_token), dtype='float32')
    h = Concatenate()([embed, static_feat_from_aa])
    h = Conv1D(100, kernel_size=3, activation='relu', padding='same')(h)

    if attention:
        h = attention_3d_block(h)

    h = CuDNNLSTM(128, return_sequences=True)(h)
    h = Dropout(rate=0.5)(h)
    h = CuDNNLSTM(128, return_sequences=False)(h)
    h = Dense(n_classes, activation='softmax')(h)

    # mdl = Model(inputs=[aa_ind, features], outputs=h)
    mdl = Model(inputs=aa_ind, outputs=h)

    optimizer = Adam(lr=LR)
    # optimizer = SGD(lr=LR)

    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return mdl


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    time_steps = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    a_probas = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probas])
    return output_attention_mul


def get_embeddings(inp):
    embed = Embedding(len(ONEHOT_M), output_dim=n_chars + 1, weights=[ONEHOT_M],
                      trainable=False, dtype='float32')(inp)

    static_embed = Embedding(STATIC_AA_TO_FEAT_M.shape[0], output_dim=STATIC_AA_TO_FEAT_M.shape[1],
                             weights=[STATIC_AA_TO_FEAT_M],
                             trainable=False, dtype='float32')(inp)
    h = Concatenate()([embed, static_embed])
    return h


def rnn_model_v2(n_classes):
    aa_ind = Input(shape=(MAXLEN,), name='aa_indice')
    h = get_embeddings(aa_ind)

    h = Conv1D(100, kernel_size=3, activation='relu', padding='same')(h)
    # conv_layers = []
    # for kernel_size in [1, 3, 5]:
    #     conv = Conv1D(20, kernel_size=kernel_size, activation='relu', padding='same')(h)
    #     conv_layers.append(conv)
    # conv_layers.append(Conv1D(20, kernel_size=7, activation='relu', padding='same')(h))
    # conv_layers.append(Conv1D(20, kernel_size=9, activation='relu', padding='same')(h))
    # h = Concatenate()(conv_layers)

    h = Dropout(rate=0.5)(h)
    h = Bidirectional(CuDNNLSTM(100, return_sequences=False))(h)
    h = Dense(n_classes, activation='softmax')(h)
    mdl = Model(inputs=aa_ind, outputs=h)

    optimizer = Adam(lr=LR)
    # optimizer = SGD(lr=LR)

    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return mdl


def transfer_model(n_classes_new, mdl_path, prev_model_output_layer='lstm_2', freeze=False):
    prev_mdl = load_model(mdl_path)
    prev_mdl.layers.pop()
    if freeze:
        for layer in prev_mdl.layers:
            layer.trainable = True

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
    flags.DEFINE_integer('max_len', default=500, help='max sequence lenght')
    flags.DEFINE_float('lr', default=1e-3, help='learning rate')
    flags.DEFINE_float('memory_fraction', default=0.4, help='memory fraction')

    flags.DEFINE_string('data_path', default='/home/pierre/riken/data/swiss/swiss_with_clans.tsv',
                        help='path to tsv data')
    flags.DEFINE_string('key_to_predict', default='clan', help='key to predict (y)')
    flags.DEFINE_string('log_dir', default='./logs', help='path to save ckpt and summaries')
    flags.DEFINE_string('transfer_path', default=None, help='path to ckpt if doing transfer learning')
    flags.DEFINE_bool('transfer_freeze', default=False, help='Should layers for last dense be froze')
    flags.DEFINE_string('layer_name', default=None, help='Name of layer to use for transfer')
    flags.DEFINE_string('groups', default='NO', help='should we use groups')
    FLAGS = flags.FLAGS

    RANDOM_STATE = 42
    MAXLEN = FLAGS.max_len
    LR = FLAGS.lr
    DATA_PATH = FLAGS.data_path
    KEY_TO_PREDICT = FLAGS.key_to_predict
    LOG_DIR = FLAGS.log_dir
    TRANSFER_PATH = FLAGS.transfer_path
    TRANSFER_FREEZE = FLAGS.transfer_freeze
    LAYER_NAME = FLAGS.layer_name
    GROUPS = FLAGS.groups if FLAGS.groups!='NO' else None
    SPLITTER = data_op.shuffle_indices if GROUPS is None else data_op.group_shuffle_indices

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
    set_session(tf.Session(config=config))

    df = pd.read_csv(DATA_PATH, sep='\t').dropna()
    # df = df.loc[df.seq_len >= 50, :]

    try:
        df.loc[:, 'sequences'] = df.sequences_x
    except:
        pass

    sequences, y = df['sequences'].values, df[KEY_TO_PREDICT]
    y = pd.get_dummies(y).values
    X = pad_sequences([[safe_char_to_idx(char) for char in sequence] for sequence in sequences], maxlen=MAXLEN)
    if GROUPS is None:
        groups = None
    else:
        groups = df[GROUPS].values

    # features = np.array([records_maker.get_feat(tokens) for tokens in X])
    train_inds, test_inds = SPLITTER(sequences, y, groups)
    print(train_inds.shape, test_inds.shape)
    Xtrain, Xtest, ytrain, ytest = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
    # features_train, features_test = features[train_inds], features[test_inds]

    if TRANSFER_PATH is None:
        model = rnn_model_v2(n_classes=y.shape[1])
        # model = rnn_model(n_classes=y.shape[1], n_features_wo_token=None, attention=False)
    else:
        model = transfer_model(mdl_path=TRANSFER_PATH, n_classes_new=y.shape[1],
                               prev_model_output_layer=LAYER_NAME, freeze=TRANSFER_FREEZE)
    print(model.summary())

    tb = TensorBoard(log_dir=LOG_DIR,
                     # histogram_freq=1,
                     # write_grads=True
                     )
    ckpt = ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                           verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(Xtrain, ytrain,  # bf [Xtrain, features_train], ...
              batch_size=128,
              epochs=100,
              validation_data=(Xtest, ytest),
              callbacks=[tb, ckpt])
