import sys
import argparse
from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from riken.nn_utils.io_tools import eval_input_fn
from riken.similarity_learning.similarity_trainer import estimator_def


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_path', type=str)
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-records_path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parser()
    CKPT_PATH = ARGS.ckpt_path
    BATCH_SIZE = ARGS.batch_size
    VAL_PATH = ARGS.records_path
    PARAMS = {
        'lstm_size': 16,
        'n_classes': 2,
        'margin': 0.5,
        # 'max_size': FLAGS.max_size,
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-2),
        'batch_size': 128
    }

    my_eval_fn = partial(eval_input_fn, path=VAL_PATH, max_size=500,
                         batch_size=BATCH_SIZE, drop_remainder=True)
    nxt = my_eval_fn()
    print(nxt)

    restored_graph = tf.Graph()
    vectors_li = []
    labels_li = []

    config_params = tf.estimator.RunConfig(model_dir=CKPT_PATH)
    mdl = estimator_def(PARAMS, cfg=config_params)
    representations = mdl.predict(my_eval_fn)
    for rep in representations:
        vectors_li.append(rep['vectors'].tolist())

    nxt = my_eval_fn()
    with tf.Session() as sess:
        while True:
            try:
                _, labels = sess.run(nxt)
                labels_li += labels.tolist()
            except tf.errors.OutOfRangeError:
                break
    print(len(labels_li))
    print(len(vectors_li))
    print(vectors_li[0])
    # sys.exit(0)
    print('done')
    labels_li = np.array(labels_li)
    vectors_li = np.array(vectors_li)
    print(labels_li.shape)
    print(vectors_li.shape)
    labels_df = pd.Series(labels_li)
    labels_df.to_csv('./export/labels.tsv', index=False, sep='\t')
    vectors_df = pd.DataFrame(vectors_li)
    vectors_df.to_csv('./export/vectors.tsv', index=False, sep='\t', header=False)

    feature_importances = mdl.get_variable_value(name='vector_representation/feature_importances/kernel')
    feature_importances = np.squeeze(feature_importances)

    # ['char_{}'.format(idx) for idx in range(24)] \
    index = ['blosum_{}'.format(idx) for idx in range(23)] \
            + ['chemprop_{}'.format(idx) for idx in range(19)] \
            + ['kidera_{}'.format(idx) for idx in range(10)]
    # + ['psiblast_{}'.format(idx) for idx in range(42)]
    pd.Series(data=feature_importances, index=index).to_csv('./export/feature_importances.csv', sep='\t')





