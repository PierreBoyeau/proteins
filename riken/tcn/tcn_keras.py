import pandas as pd
from keras.layers import Activation, Add, BatchNormalization, Conv1D, Dense, Dropout, Input, Lambda, RepeatVector, Permute, Multiply
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K
from tensorflow import flags
import os
from riken.protein_io import data_op
from riken.rnn.rnn_keras import get_embeddings, safe_char_to_idx, transfer_model

from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

"""
python tcn_keras.py \
max_len 1000 \
lr 1e-3
log_dir logs_v1_swisstrain


python tcn_keras.py \
-max_len 1000 \
-lr 1e-3 \
-layer_name lambda_2 \
-data_path /home/pierre/riken/data/riken_data/complete_from_xlsx.tsv \
-key_to_predict is_allergenic \
-log_dir logs_transfer_second_try \
-groups species \

"""


def residual_block(input, dilatation, kernel_size, n_filters, dropout_rate, do1conv=True):
    conv = input
    for _ in range(2):
        conv = Conv1D(n_filters, kernel_size=kernel_size, dilation_rate=dilatation, padding='causal')(conv)
        # here do weight norm (later)
        # instead here use of batch norm because already implemented
        conv = BatchNormalization()(conv)
        conv = Activation(activation='relu')(conv)
        conv = Dropout(rate=dropout_rate)(conv)

    if do1conv:
        rescaled_input = Conv1D(n_filters, kernel_size=1)(input)
    else:
        rescaled_input = input

    last = Add()([conv, rescaled_input])
    return last


def tcn_model(n_classes, depth, n_filters, kernel_size, dropout_rate=0.0):
    inp = Input(shape=(MAXLEN,))
    h = get_embeddings(inp=inp)

    # h = inp

    for it in range(depth):
        do1conv = (it == 0)  # 1conv done only first layer (elsewhere number of filters stays the same
        h = residual_block(h, dilatation=2**it, n_filters=n_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                           do1conv=do1conv)

    # h = h[:, -1, :]
    # h = Lambda(lambda x: x[:, -1, :])(h)

    attention = Dense(1)(h)
    attention = Lambda(lambda x: K.squeeze(x, axis=2))(attention)
    attention = Activation(activation='softmax')(attention)
    attention = RepeatVector(n_filters)(attention)
    attention = Permute((2, 1))(attention)
    last = Multiply()([attention, h])
    last = Lambda(lambda x: K.sum(x, axis=1), output_shape=(n_filters,))(last)

    h = Dense(n_classes, activation='softmax')(last)
    mdl = Model(inputs=inp, outputs=h)

    # optimizer = Adam(lr=LR)
    optimizer = RMSprop(lr=LR)
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    return mdl


if __name__ == '__main__':
    flags.DEFINE_integer('max_len', default=1000, help='max sequence lenght')
    flags.DEFINE_float('lr', default=1e-3, help='learning rate')
    flags.DEFINE_string('data_path', default='/home/pierre/riken/data/swiss/swiss_with_clans.tsv',
                        help='path to tsv data')
    flags.DEFINE_string('key_to_predict', default='clan', help='key to predict (y)')
    flags.DEFINE_string('log_dir', default='./logs', help='path to save ckpt and summaries')
    flags.DEFINE_string('transfer_path', default=None, help='path to ckpt if doing transfer learning')
    flags.DEFINE_bool('transfer_freeze', default=False, help='Should layers for last dense be froze')
    flags.DEFINE_string('layer_name', default=None, help='Name of layer to use for transfer')
    flags.DEFINE_string('groups', default='NO', help='should we use groups')
    flags.DEFINE_integer('kernel_size', default=3, help='kernel size')
    flags.DEFINE_integer('nb_filters', default=50, help='nb_filters')
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
    KERNEL_SIZE = FLAGS.kernel_size
    NB_FILTERS = FLAGS.nb_filters

    df = pd.read_csv(DATA_PATH, sep='\t').dropna()
    # df = df.loc[df.seq_len >= 50, :]

    try:
        df.loc[:, 'sequences'] = df.sequences_x
    except AttributeError:
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
        # model = rnn_model_v2(n_classes=y.shape[1])
        model = tcn_model(n_classes=y.shape[1], depth=8, n_filters=NB_FILTERS, kernel_size=KERNEL_SIZE, dropout_rate=0.5)
    else:
        model = transfer_model(n_classes_new=y.shape[1], mdl_path=TRANSFER_PATH, prev_model_output_layer='lambda_1')
    print(model.summary())

    tb = TensorBoard(log_dir=LOG_DIR)
    ckpt = ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                           verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(Xtrain, ytrain,  # bf [Xtrain, features_train], ...
              batch_size=128,
              epochs=100,
              validation_data=(Xtest, ytest),
              callbacks=[tb, ckpt])
    print(roc_auc_score(ytest[:, 1], model.predict(Xtest)[:, 1]))
