from functools import partial

import tensorflow as tf
from tensorflow import flags

from riken.nn_utils.io_tools import train_input_fn, eval_input_fn
from riken.similarity_learning import similarity_model


def model_fn(features, labels, mode=None, params=None, config=None):
    print(features, labels)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predict_model = similarity_model.SimilarityModel(inputs=features, labels=labels,
                                                         predict=True, **params)
        return tf.estimator.EstimatorSpec(mode,
                                          predictions={"vectors": predict_model.vectors})

    model = similarity_model.SimilarityModel(inputs=features, labels=labels, **params)
    model.build()
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=model.loss)
    assert mode == tf.estimator.ModeKeys.TRAIN
    train_op = model.optimizer
    return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op)


def estimator_def(parameters, cfg):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=parameters, config=cfg)


def transfer_model(parameters, cfg, transfer_path):
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=transfer_path,
                                        vars_to_warm_start='.*transferable.*')
    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=parameters, config=cfg, warm_start_from=ws)


def parse_args():
    flags.DEFINE_string('train_path',
                        '/home/pierre/riken/riken/rnn/records/train_riken_data.tfrecords',
                        'Path to training records')
    flags.DEFINE_string('val_path',
                        '/home/pierre/riken/riken/rnn/records/test_riken_data.tfrecords',
                        'Path to training records')

    flags.DEFINE_integer('n_classes', 590, 'Number of classes')
    flags.DEFINE_string('log_dir', './results', 'Path to training records')
    flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model on')
    flags.DEFINE_integer('batch_size', 128, 'Number of epochs to train the model on')
    flags.DEFINE_float('lr', 1e-3, 'Maximum sequence lenght')
    flags.DEFINE_integer('max_size', 500, 'max size')
    flags.DEFINE_integer('lstm_size', 4, 'max size')
    flags.DEFINE_float('margin', 1.0, 'Maximum sequence lenght')
    return flags.FLAGS


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parse_args()
    SAVE_EVERY = 60

    pssm_nb_examples = 42
    train_params = {'lstm_size': FLAGS.lstm_size,
                    'n_classes': FLAGS.n_classes,
                    'margin': FLAGS.margin,
                    # 'max_size': FLAGS.max_size,
                    'optimizer': tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
                    'batch_size': FLAGS.batch_size
                    }

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config_params = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, log_step_count_steps=10,
                                           keep_checkpoint_max=100, save_checkpoints_secs=SAVE_EVERY,
                                           session_config=sess_config)
    mdl = estimator_def(train_params, cfg=config_params)
    my_train_fn = partial(train_input_fn, path=FLAGS.train_path, max_size=FLAGS.max_size,
                          epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, drop_remainder=True)
    train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn)
    my_eval_fn = partial(eval_input_fn, path=FLAGS.val_path, max_size=FLAGS.max_size,
                         batch_size=FLAGS.batch_size, drop_remainder=True)
    eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=30,
                                      throttle_secs=SAVE_EVERY)
    # tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)

    # hook = tf_debug.TensorBoardDebugHook("griffin1:6009")
    mdl.train(my_train_fn,
              # hooks=[hook]
              )
