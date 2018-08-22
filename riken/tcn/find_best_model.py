import time

import pandas as pd
from functools import partial
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorboard.plugins.beholder import BeholderHook

from riken.nn_utils.io_tools import train_input_fn, eval_input_fn
from riken.tcn import tcn_model

"""
python trainer.py \
-train_path ./records/swiss_train_data500.tfrecords \
-val_path ./records/swiss_val_data500.tfrecords \
-log_dir ./swisstrain_psiblast_1 \
-lr 1e-3 \
-n_classes 590 \
1

python trainer.py -train_path ./records/train_riken_data.tfrecords -val_path ./records/test_riken_data.tfrecords \
-log_dir ./test3 -lr 1e-3

"""

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_string('train_path',
                    '/home/pierre/riken/riken/rnn/records/train_riken_data.tfrecords',
                    'Path to training records')
flags.DEFINE_string('val_path',
                    '/home/pierre/riken/riken/rnn/records/test_riken_data.tfrecords',
                    'Path to training records')

flags.DEFINE_integer('n_classes', 2, 'Number of classes')
FLAGS = flags.FLAGS

SAVE_EVERY = 30

pssm_nb_examples = 42

random_uniform_001 = lambda: tf.initializers.random_uniform(minval=-1e-2, maxval=1e-2)
random_uniform_01 = lambda: tf.initializers.random_uniform(minval=-1e-1, maxval=1e-1)
random_uniform_1 = lambda: tf.initializers.random_uniform(minval=-1, maxval=1)
truncated_normal_001 = lambda: tf.initializers.truncated_normal(stddev=1e-2)
truncated_normal_01 = lambda: tf.initializers.truncated_normal(stddev=1e-1)
truncated_normal_1 = lambda: tf.initializers.truncated_normal(stddev=1.0)

train_params_grid = {
    "n_classes": [2],
    "max_size": [500],
    "depth": [4, 5, 6, 7, 8, 9],
    "kernel_size": [3, 5, 7, 9, 11],
    "n_filters": [25, 50, 100, 200, 300],
    "dropout_rate": [0.1, 0.2, 0.3, 0.5],
    "optimizer": [tf.train.AdamOptimizer(learning_rate=1e-3),
                  tf.train.RMSPropOptimizer(learning_rate=1e-3),
                  tf.train.GradientDescentOptimizer(learning_rate=1e-3)],
    'conv_initializer': [
        None,
        random_uniform_001(),
        random_uniform_01(),
        random_uniform_1(),
        truncated_normal_001(),
        truncated_normal_01(),
        truncated_normal_1(),
    ],
    "batch_size": [16, 32, 64, 128]

}


def model_fn(features, labels, mode=None, params=None, config=None):
    print(features, labels)
    model = tcn_model.TCNModel(input=features['tokens'], pssm_input=features['pssm_li'],
                               labels=labels, **params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = model.probabilities
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    model.build()
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=model.loss, eval_metric_ops={'accuracy': model.acc, 'auc': model.auc})
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    train_op = model.optimize

    return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op,
                                      eval_metric_ops={'accuracy': model.acc, 'auc': model.auc})


def estimator_def(parameters, cfg):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=parameters, config=cfg)


def transfer_model(parameters, cfg, transfer_path):
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=transfer_path,
                                        vars_to_warm_start='.*transferable.*')
    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=parameters, config=cfg, warm_start_from=ws)


if __name__ == '__main__':
    from sklearn.model_selection import ParameterGrid
    from random import shuffle

    parameter_grid = list(ParameterGrid(train_params_grid))
    shuffle(parameter_grid)

    idx_to_params = dict()

    for idx, params in enumerate(parameter_grid):
        try:
            model_dir = 'hyperparameters_tune/{}'.format(idx)
            idx_to_params[idx] = params
            pd.DataFrame(idx_to_params).to_csv('mappings.tsv', sep='\t')

            params_cp = params.copy()
            batch_size = params_cp.pop('batch_size')

            my_train_fn = partial(train_input_fn, path=FLAGS.train_path, max_size=500,
                                  epochs=10, batch_size=batch_size)
            train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn, max_steps=5000)
            my_eval_fn = partial(eval_input_fn, path=FLAGS.val_path, max_size=500,
                                 batch_size=batch_size)
            eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=30,
                                              throttle_secs=60)

            config_params = tf.estimator.RunConfig(model_dir=model_dir,
                                                   keep_checkpoint_max=2,
                                                   save_checkpoints_secs=SAVE_EVERY)
            mdl = estimator_def(params_cp, cfg=config_params)
            tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
        except:
            print('ERROR ...')
            # raise
            time.sleep(5)
            print('RESUMING ...')
            continue
