import tensorflow as tf

from riken.rnn import rnn_model
import time
"""
python trainer.py \
-train_path /home/pierre/train_data.tfrecords \
-val_path /home/pierre/val_data.tfrecords \
-log_dir ./logs_pssm_test \
-lr 1e-3 \
-n_classes 598 \
-lstm_size_list [128]
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
flags.DEFINE_string('log_dir', './results', 'Path to training records')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model on')
flags.DEFINE_integer('batch_size', 128, 'Number of epochs to train the model on')
flags.DEFINE_boolean('do_train', False, 'do train')
flags.DEFINE_boolean('do_eval', False, 'do eval')
flags.DEFINE_float('lr', 1e-3, 'Maximum sequence lenght')
FLAGS = flags.FLAGS

SAVE_EVERY = 600

assert FLAGS.do_eval == (not FLAGS.do_train)

pssm_nb_examples = 42
train_params = {'lstm_size': 128,
                'n_classes': 2,
                'max_size': 500,
                'dropout_keep_p': 0.3,
                'optimizer': tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
                'conv_n_filters': 100}


def _parse_function(example_proto):
    features = {
        'sentence_len': tf.FixedLenFeature((), tf.int64, default_value=0),
        'tokens': tf.FixedLenFeature([train_params['max_size']], tf.int64),
        # 'pssm_li': tf.FixedLenFeature([train_params['max_size']*42], tf.float32,
        #                               default_value=0),
        # 'pssm_li': tf.VarLenFeature(tf.float32),
        'pssm_li': tf.FixedLenFeature((train_params['max_size']*pssm_nb_examples), tf.float32),
        'n_features_pssm': tf.FixedLenFeature((), tf.int64, default_value=0),
        'label': tf.FixedLenFeature((), tf.int64, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['pssm_li'] = tf.reshape(parsed_features['pssm_li'],
                                            shape=(train_params['max_size'], pssm_nb_examples))
    labels = parsed_features.pop('label')
    return parsed_features, labels


def input_fn(path, epochs):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(count=epochs)
    batched_dataset = dataset.batch(batch_size=FLAGS.batch_size)
    iterator = batched_dataset.make_one_shot_iterator()
    nxt = iterator.get_next()
    return nxt


def train_input_fn():
    return input_fn(FLAGS.train_path, epochs=FLAGS.epochs)


def eval_input_fn():
    return input_fn(FLAGS.val_path, epochs=1)


def estimator_def(parameters, config):
    def model_fn(features, labels, mode=None, params=None, config=None):
        model = rnn_model.RnnModel(input=features['tokens'], pssm_input=features['pssm_li'],
                                   labels=labels, **params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = model.probabilities
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=model.loss, eval_metric_ops={'accuracy': model.acc, 'auc': model.auc})
        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN
        train_op = model.optimize

        return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op,
                                          eval_metric_ops={'accuracy': model.acc, 'auc': model.auc})

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=parameters, config=config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, log_step_count_steps=10, keep_checkpoint_max=100,
                                    save_checkpoints_secs=SAVE_EVERY, session_config=sess_config)
    mdl = estimator_def(train_params, config=config)

    # if FLAGS.do_train:
    #     print('TRAINING MODE ...\n')
    #     mdl.train(train_input_fn)
    # elif FLAGS.do_eval:
    #     print('EVAL MODE ...\n')
    #     try:
    #         while True:
    #             mdl.evaluate(eval_input_fn)
    #             time.sleep(SAVE_EVERY)
    #     except KeyboardInterrupt:
    #         print('Exiting validation ...')

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=30, throttle_secs=60)
    tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
