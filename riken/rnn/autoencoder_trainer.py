from functools import partial
import argparse
import tensorflow as tf
from riken.nn_utils.io_tools import train_input_fn, eval_input_fn
from riken.rnn import rnn_model


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
    # loss = tf.contrib.seq2seq.sequence_loss(logits=decoder.logits,
    # targets=features['tokens'], average_across_timestep=True)
    element_wise_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=features["tokens"],
                                                                   logits=decoder.logits,
                                                                   dim=-1)
    n_zeros = tf.equal(features['tokens'], 0)
    n_zeros = tf.reduce_sum(n_zeros, axis=-1)
    err_loss = tf.reduce_mean(element_wise_loss / n_zeros)
    kl_loss = -0.5*tf.reduce_mean(1.0
                                  + decoder.log_sgm
                                  - tf.sqrt(decoder.means)
                                  - tf.exp(decoder.log_sgm))
    loss = err_loss + kl_loss
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
    FLAGS = parse_arguments()
    SAVE_EVERY = 60
    AE_PARAMS = {
        'batch_size': 64,
        'epochs': 50,
        'max_size': 500,
        'encoder': {
            'lstm_size': 16,
            'n_classes': 2,
            'max_size': 500,
            'dropout_keep_p': 0.25,
            'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
            'conv_n_filters': 25,
            'two_lstm_layers': False,
        },
        'decoder': {},
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3)
    }

    config_params = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, log_step_count_steps=10,
                                           keep_checkpoint_max=100, save_checkpoints_secs=SAVE_EVERY)
    mdl = autoencoder_def(AE_PARAMS, cfg=config_params)

    my_train_fn = partial(train_input_fn, path=FLAGS.train_path, max_size=AE_PARAMS['max_size'],
                          epochs=AE_PARAMS['epochs'], batch_size=AE_PARAMS['batch_size'])
    train_spec = tf.estimator.TrainSpec(input_fn=my_train_fn)
    my_eval_fn = partial(eval_input_fn, path=FLAGS.val_path, max_size=AE_PARAMS['max_size'],
                         batch_size=AE_PARAMS['batch_size'])
    eval_spec = tf.estimator.EvalSpec(input_fn=my_eval_fn, start_delay_secs=30,
                                      throttle_secs=SAVE_EVERY)
    tf.estimator.train_and_evaluate(mdl, train_spec, eval_spec)
