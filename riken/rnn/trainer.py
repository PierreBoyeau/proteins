from functools import partial
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorboard.plugins.beholder import BeholderHook

from riken.nn_utils.io_tools import train_input_fn, eval_input_fn
from riken.rnn import rnn_model

"""
python trainer.py \
-train_path ./records/swiss_train_data500.tfrecords \
-val_path ./records/swiss_val_data500.tfrecords \
-log_dir ./swisstrain_psiblast_1 \
-lr 1e-3 \
-n_classes 590 \


python trainer.py -train_path ./records/train_riken_data.tfrecords -val_path ./records/test_riken_data.tfrecords \
-log_dir ./test3 -lr 1e-3

"""

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_string('train_path',
                    # '/home/pierre/riken/riken/rnn/records/train_swiss_with_pssm.tfrecords',
                    '/home/pierre/riken/riken/rnn/records/train_riken_data.tfrecords',
                    'Path to training records')
flags.DEFINE_string('val_path',
                    # '/home/pierre/riken/riken/rnn/records/val_swiss_with_pssm.tfrecords',
                    '/home/pierre/riken/riken/rnn/records/test_riken_data.tfrecords',
                    'Path to training records')

flags.DEFINE_integer('n_classes', 590, 'Number of classes')

flags.DEFINE_string('transfer_path', None, 'path to ckpt if we want to do transfer learning')

flags.DEFINE_string('log_dir', './results', 'Path to training records')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model on')
flags.DEFINE_integer('batch_size', 128, 'Number of epochs to train the model on')
flags.DEFINE_float('lr', 1e-3, 'Maximum sequence lenght')
flags.DEFINE_integer('max_size', 500, 'max size')

flags.DEFINE_bool('debug', False, 'use debugger')

FLAGS = flags.FLAGS

SAVE_EVERY = 600

pssm_nb_examples = 42
train_params = {'lstm_size': 128,
                'n_classes': FLAGS.n_classes,
                'max_size': FLAGS.max_size,
                'dropout_keep_p': 0.5,
                'optimizer': tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
                'conv_n_filters': 100,
                'two_lstm_layers': True}


def model_fn(features, labels, mode=None, params=None, config=None):
    print(features, labels)
    model = rnn_model.RnnModel(input=features['tokens'], pssm_input=features['pssm_li'], labels=labels, **params)
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
    tf.logging.set_verbosity(tf.logging.INFO)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config_params = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, log_step_count_steps=10,
                                           keep_checkpoint_max=100, save_checkpoints_secs=SAVE_EVERY,
                                           session_config=sess_config)
    if FLAGS.transfer_path is None:
        mdl = estimator_def(train_params, cfg=config_params)
    else:
        mdl = transfer_model(train_params, cfg=config_params, transfer_path=FLAGS.transfer_path)

    # beholder_hook = BeholderHook(FLAGS.log_dir)

    my_train_fn = partial(train_input_fn, path=FLAGS.train_path, max_size=FLAGS.max_size,
                          epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
    train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn)
    my_eval_fn = partial(eval_input_fn, path=FLAGS.val_path, max_size=FLAGS.max_size,
                         batch_size=FLAGS.batch_size)
    eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=30, throttle_secs=600)

    if FLAGS.debug:
        debug_hk = tf_debug.TensorBoardDebugHook("griffin1:6009")
        # mdl.train(my_train_fn, hooks=[debug_hk])
        mdl.evaluate(my_eval_fn, hooks=[debug_hk])
    else:
        tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
