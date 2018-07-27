import os
import tensorflow as tf

import rnn_model

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags

flags.DEFINE_string('train_path', '/home/pierre/riken/rnn/train_data.tfrecords', 'Path to training records')
flags.DEFINE_string('val_path', '/home/pierre/riken/rnn/val_data.tfrecords', 'Path to training records')

flags.DEFINE_string('log_dir', '/home/pierre/riken/rnn/results', 'Path to training records')
flags.DEFINE_float('dropout_keep_p', 0.75, 'Keep proba dropout')

flags.DEFINE_integer('epochs', 10, 'Number of epochs to train the model on')
flags.DEFINE_integer('batch_size', 128, 'Batch Size')
flags.DEFINE_float('lr', 1e-2, 'Maximum sequence lenght')
flags.DEFINE_integer('max_len', 500, 'Maximum sequence lenght')

flags.DEFINE_integer('n_classes', 0, help='Number of classes')
flags.DEFINE_integer('eval_batches', 128, 'Number of batches to process every eval period')
flags.DEFINE_integer('eval_every_n_iter', 1000, 'How frequent should eval be done')
flags.DEFINE_list('lstm_size_list', [128, 128], 'List of LSTM cells')
FLAGS = flags.FLAGS


def _parse_function(example_proto):
    features = {
        'sentence_len': tf.FixedLenFeature((), tf.int64, default_value=0),
        # 'sentence': tf.FixedLenFeature((), tf.string, default_value=""),
        'tokens': tf.FixedLenFeature([FLAGS.max_len], tf.int64),

        # 'sentence': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature((), tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["sentence_len"], parsed_features["tokens"], parsed_features["label"]


def train_input_fn():
    dataset = tf.data.TFRecordDataset(FLAGS.train_path)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(count=FLAGS.epochs)
    batched_dataset = dataset.batch(batch_size=FLAGS.batch_size)
    iterator = batched_dataset.make_one_shot_iterator()
    next = iterator.get_next()
    sentences_len, tokens, labels = next

    return tokens, labels


def eval_input_fn():
    dataset_eval = tf.data.TFRecordDataset(FLAGS.val_path)
    dataset_eval = dataset_eval.map(_parse_function)
    batched_dataset_eval = dataset_eval.batch(batch_size=FLAGS.batch_size)
    iterator_eval = batched_dataset_eval.make_one_shot_iterator()
    next_eval = iterator_eval.get_next()
    _, tokens_eval, labels_eval = next_eval

    return tokens_eval, labels_eval


def manual_train():
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

    tokens, labels = train_input_fn()
    # tokens_eval, labels_eval = eval_input_fn()
    with tf.variable_scope('model'):
        model_train = rnn_model.RnnModel(lstm_size_list=FLAGS.lstm_size_list, n_classes=FLAGS.n_classes,
                                         vocab_size=25, learning_rate=FLAGS.lr,
                                         max_size=FLAGS.max_len, embed_dim=10, dropout_keep_p=FLAGS.dropout_keep_p,
                                         optimizer=optimizer)

    # with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #     model_eval = rnn_model.RnnModel(input=tokens_eval, labels=labels_eval,
    #                                     lstm_size_list=FLAGS.lstm_size_list, n_classes=FLAGS.n_classes,
    #                                     vocab_size=25, learning_rate=FLAGS.lr,
    #                                     max_size=FLAGS.max_len, embed_dim=10, dropout_keep_p=1.0)
    opt = model_train.optimize

    loss_summ = tf.summary.scalar('loss', model_train.loss)
    acc_summ = tf.summary.scalar('accuracy', model_train.acc)

    tf.summary.histogram('predictions', tf.argmax(model_train.logits, 1))
    # train_summary_op = tf.summary.merge_all()
    # tf.summary.scalar('eval/loss', model_eval.loss)
    # tf.summary.scalar('eval/auc', model_eval.auc)
    merger = tf.summary.merge_all()

    logtensors = {
        "step": tf.train.get_or_create_global_step(),
        "train_loss": model_train.loss,
        "train_acc": model_train.acc
    }

    hks = [
        tf.train.SummarySaverHook(
            save_steps=500,
            summary_op=merger,
            output_dir=os.path.join(FLAGS.log_dir, 'summaries')
        ),
        tf.train.CheckpointSaverHook(
            FLAGS.log_dir,
            save_secs=60 * 10
        ),
        tf.train.LoggingTensorHook(
            logtensors,
            every_n_iter=1,
        )
    ]
    step = tf.train.get_or_create_global_step(graph=None)
    increment_op = tf.assign(step, step+1)
    num_batch = 0
    print(tokens.get_shape())
    with tf.train.MonitoredTrainingSession(hooks=hks, checkpoint_dir=FLAGS.log_dir) as sess:
        while True:
            try:
                # sentences_len, tokens, labels = sess.run(next)
                _ = sess.run([opt, increment_op])
                num_batch += 1

            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    manual_train()
    # convo_train()



# def lstm_model_fn(features, labels, mode, params):
#     mdl = rnn_model.RnnModel(input=features, labels=labels, **params)
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         predictions = {'probabilities': mdl.logits}
#         return tf.estimator.EstimatorSpec(mode, predictions=predictions)
#
#     metrics = {'auc': mdl.auc}
#     if mode == tf.estimator.ModeKeys.EVAL:
#         return tf.estimator.EstimatorSpec(mode, loss=mdl.loss, eval_metric_ops=metrics)
#
#     assert mode == tf.estimator.ModeKeys.TRAIN
#     return tf.estimator.EstimatorSpec(mode, loss=mdl.loss, train_op=mdl.optimize)
