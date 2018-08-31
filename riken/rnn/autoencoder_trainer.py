from functools import partial
import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from riken.nn_utils.io_tools import train_input_fn, eval_input_fn
from riken.protein_io.prot_features import chars
from riken.rnn import rnn_model

"""
python autoencoder_trainer.py -train_path records/riken_data_v2_l1000_post_train.tfrecords \
-val_path records/riken_data_v2_l1000_post_test.tfrecords \
-log_dir ae_test
"""


def autoencoder_fn(features, labels, mode=None, params=None, config=None):
    print(features, labels)
    encoder_params = params['encoder']
    decoder_params = params['decoder']
    optim = params['optimizer']

    with tf.variable_scope('encoder'):
        encoder_model = rnn_model.RnnModel(input=features['tokens'], pssm_input=features['pssm_li'],
                                           labels=labels, **encoder_params)
        encoder = encoder_model.attention_output
    with tf.variable_scope('decoder'):
        decoder = rnn_model.RnnDecoder(encoder_input=encoder, **decoder_params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = decoder.predictions
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    input_lbl = features['tokens']
    input_lbl = tf.maximum(input_lbl, 0)

    # V1
    n_chars = tf.cast(tf.not_equal(input_lbl, 0), tf.float32)
    # tokens_without_negs = tf.
    loss = tf.contrib.seq2seq.sequence_loss(logits=decoder.logits,
                                            targets=input_lbl,
                                            weights=n_chars)
    # V2
    # targets=features['tokens'], average_across_timestep=True)
    # onehot_labels = tf.one_hot(input_lbl, depth=len(chars)+1)
    # element_wise_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
    #                                                                logits=decoder.logits,
    #                                                                dim=-1,
    #                                                                name='element_wise_loss')
    # _, n_feat = element_wise_loss.get_shape().as_list()
    # err_loss = tf.reduce_mean(element_wise_loss, name='error_loss')
    # loss = err_loss
    # kl_loss = -0.5*tf.reduce_mean(1.0
    #                               + decoder.log_sgm
    #                               - tf.square(decoder.means)
    #                               - tf.exp(decoder.log_sgm), name='kl_loss')
    # loss = err_loss + kl_loss

    mapping_string = tf.constant(chars)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping_string, default_value="<>")
    input_strings = tf.reduce_join(table.lookup(input_lbl-1), axis=1)
    output_strings = tf.reduce_join(table.lookup(decoder.predictions-1), axis=1)

    tf.summary.text('input_strings', input_strings[0])
    tf.summary.text('output_strings', output_strings[0])

    tf.summary.histogram('output_indices', input_lbl-1)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)
    assert mode == tf.estimator.ModeKeys.TRAIN
    train_op = optim.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def autoencoder_def(parameters, cfg):
    return tf.estimator.Estimator(model_fn=autoencoder_fn,
                                  params=parameters, config=cfg)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', type=str, help='Path to train record')
    parser.add_argument('-val_path', type=str, help='Path to val record')
    parser.add_argument('-log_dir', type=str, help='directory where summaries/ckpt will be saved')
    return parser.parse_args()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_arguments()
    SAVE_EVERY = 500
    AE_PARAMS = {
        'batch_size': 128,
        'epochs': 200,
        'max_size': 1000,
        'encoder': {
            'lstm_size': 16,
            'n_classes': 2,
            'max_size': 1000,
            'dropout_keep_p': 0.25,
            'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
            'conv_n_filters': 25,
            'two_lstm_layers': False,
        },
        'decoder': {
            'max_size': 1000,
            'lstm_size': 16,
            'n_hidden': 200,
        },
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-1)
    }

    config_params = tf.estimator.RunConfig(model_dir=args.log_dir, log_step_count_steps=10,
                                           keep_checkpoint_max=100, save_checkpoints_secs=SAVE_EVERY)
    mdl = autoencoder_def(AE_PARAMS, cfg=config_params)

    my_train_fn = partial(train_input_fn, path=args.train_path, max_size=AE_PARAMS['max_size'],
                          epochs=AE_PARAMS['epochs'], batch_size=AE_PARAMS['batch_size'])
    train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn)
    my_eval_fn = partial(eval_input_fn, path=args.val_path, max_size=AE_PARAMS['max_size'],
                         batch_size=AE_PARAMS['batch_size'])
    eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=30,
                                      throttle_secs=SAVE_EVERY)

    # debug_hk = tf_debug.TensorBoardDebugHook("griffin1:6009")
    # mdl.train(my_train_fn, hooks=[debug_hk])
    # tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
    mdl.train(my_train_fn)
