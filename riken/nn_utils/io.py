import tensorflow as tf

"""
Scripts useful to READ tensorflow records and to create tf.Dataset objects easily.
"""


def _parse_function(example_proto, params, pssm_nb_f=42):
    """
    Returns tensors to feed to tensorflow models
    :param example_proto:
    :param params: dict giving shapes of objects that need to be read
    :param pssm_nb_f: number of PSSM features (42 if using PSIBLAST)
    :return: features, labels
    """
    features = {
        'sentence_len': tf.FixedLenFeature((), tf.int64, default_value=0),
        'tokens': tf.FixedLenFeature([params['max_size']], tf.int64),
        # 'pssm_li': tf.FixedLenFeature([train_params['max_size']*42], tf.float32,
        #                               default_value=0),
        # 'pssm_li': tf.VarLenFeature(tf.float32),
        'pssm_li': tf.FixedLenFeature((params['max_size']*pssm_nb_f), tf.float32),
        'n_features_pssm': tf.FixedLenFeature((), tf.int64, default_value=0),
        'label': tf.FixedLenFeature((), tf.int64, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['pssm_li'] = tf.reshape(parsed_features['pssm_li'],
                                            shape=(params['max_size'], pssm_nb_f))
    labels = parsed_features.pop('label')
    return parsed_features, labels


def input_fn(path, epochs, shuffle=True):
    """
    Create tf.Dataset object for training tf model
    :param path: path to records
    :param epochs: nb of epochs
    :param shuffle: bool: should we shuffle
    :return: iterator
    """
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(count=epochs)
    batched_dataset = dataset.batch(batch_size=FLAGS.batch_size)
    iterator = batched_dataset.make_one_shot_iterator()
    nxt = iterator.get_next()
    return nxt


def train_input_fn():
    return input_fn(FLAGS.train_path, epochs=FLAGS.epochs)


def eval_input_fn():
    return input_fn(FLAGS.val_path, epochs=1, shuffle=False)