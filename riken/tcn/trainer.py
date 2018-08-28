from functools import partial
import tensorflow as tf
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
flags.DEFINE_string('log_dir', './results', 'Path to training records')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model on')
flags.DEFINE_integer('batch_size', 128, 'Number of epochs to train the model on')
flags.DEFINE_float('lr', 1e-3, 'Maximum sequence lenght')
flags.DEFINE_integer('max_size', 500, 'max size')
FLAGS = flags.FLAGS
SAVE_EVERY = 60

pssm_nb_examples = 42
train_params = {'depth': 8,
                'n_classes': FLAGS.n_classes,
                'max_size': FLAGS.max_size,
                'kernel_size': 7,
                'dropout_rate': 0.25,
                'optimizer': tf.train.RMSPropOptimizer(learning_rate=1e-3),
                'n_filters': 25}


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


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config_params = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, log_step_count_steps=10,
                                           keep_checkpoint_max=100, save_checkpoints_secs=SAVE_EVERY,
                                           session_config=sess_config)
    mdl = estimator_def(train_params, cfg=config_params)
    beholder_hook = BeholderHook(FLAGS.log_dir)
    my_train_fn = partial(train_input_fn, path=FLAGS.train_path, max_size=FLAGS.max_size,
                          epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
    train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn, hooks=[beholder_hook])
    my_eval_fn = partial(eval_input_fn, path=FLAGS.val_path, max_size=FLAGS.max_size,
                         batch_size=FLAGS.batch_size)
    eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=SAVE_EVERY,
                                      throttle_secs=30)
    tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
