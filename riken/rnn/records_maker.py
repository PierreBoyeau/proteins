import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from riken.prot_features.prot_features import chars, chars_to_idx
from riken.prot_features import prot_features

flags = tf.flags

MAX_LEN = 500
RANDOM_STATE = 42
VALUE = -1


def safe_char_to_idx(char):
    if char in chars_to_idx:
        return chars_to_idx[char]
    else:
        return


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_feat(int_seq_tokens):
    aa_to_feat = prot_features.get_blosum80_dict_to_features()
    feat_len = len(aa_to_feat['A'])
    sequence_features = []
    for ind in int_seq_tokens:
        if ind <= 0:
            feat = np.zeros(feat_len)
        else:
            try:
                char = chars[ind - 1]
                feat = np.array(aa_to_feat[char])
            except KeyError:
                feat = np.zeros(feat_len)
        sequence_features.append(feat)
    return np.array(sequence_features)


def write_record(my_df, record_path, y_tag, pssm_format_fi='../data/psiblast/swiss/{}_pssm.txt'):
    sequences, y, indices = my_df['sequences'].values, my_df[y_tag].astype('category'), my_df.index.values
    writer = tf.python_io.TFRecordWriter(record_path)
    for sen, label_id, id in zip(tqdm(sequences), y.cat.codes, indices):
        pssm_path = pssm_format_fi.format(id)
        pssm = pd.read_csv(pssm_path, sep=' ', skiprows=2, skipfooter=6, skipinitialspace=True)\
            .reset_index(level=[2, 3])
        pssm_feat = pssm.iloc[:MAX_LEN].values
        seq_len, n_features_pssm = pssm_feat.shape
        pssm_mat = np.zeros(shape=(MAX_LEN, n_features_pssm))
        pssm_mat[-seq_len:] = pssm_feat
        pssm_mat = pssm_mat.reshape(-1)

        # if seq_len != len(sen):
        #     print('Inconsistency for protein id : {}'.format(id))
        #     print(sen)
        #     print(pssm.index.values)

        tokens = [char for char in sen]
        tokens = np.array([safe_char_to_idx(char) for char in tokens])
        padded_tokens = pad_sequences(tokens.reshape(1, -1), maxlen=MAX_LEN, value=VALUE).reshape(-1)
        # padded_blosum_feat = get_feat(padded_tokens)
        feature = {
            'sentence_len': _int64_feature([len(sen)]),
            # 'sentence': _byte_feature(str.encode(sen)),
            'tokens': _int64_feature(padded_tokens),
            'pssm_li': _float_feature(pssm_mat),
            'n_features_pssm': _int64_feature([n_features_pssm]),
            # 'blosum_feat': _float_feature(padded_blosum_feat),
            'label': _int64_feature([label_id])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    flags.DEFINE_string('train_path', './swiss_train_data500.tfrecords', 'Path of training records')
    flags.DEFINE_string('val_path', './swiss_val_data500.tfrecords', 'Path of val records')
    FLAGS = flags.FLAGS

    train_records_filename = FLAGS.train_path
    val_records_filename = FLAGS.val_path

    df = pd.read_csv('/home/pierre/riken/data/swiss/swiss_with_clans.tsv', sep='\t')
    y_name = 'clan'
    df.loc[:, 'sequences'] = df.sequences_x
    train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=0.2)

    pssm_format_file = '../../data/psiblast/swiss/{}_pssm.txt'
    # Writing Train data
    write_record(train_df, train_records_filename, y_tag=y_name, pssm_format_fi=pssm_format_file)
    # Writing Val data
    write_record(val_df, val_records_filename, y_tag=y_name, pssm_format_fi=pssm_format_file)

    print('Nb classes: ', len(np.unique(df)))
