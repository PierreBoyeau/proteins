import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from riken.protein_io import prot_features, data_op

flags = tf.flags

MAX_LEN = 500
RANDOM_STATE = 42
VALUE = -1

"""
Script that can be use to WRITE tensorflow records
"""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_feat(int_seq_tokens):
    """
    DO NOT USE WITH static amino acid features (if this is the case, just access these features inside computation graph)
    :param int_seq_tokens:
    :return:
    """
    aa_to_feat = prot_features.get_blosum80_dict_to_features()
    feat_len = len(aa_to_feat['A'])
    sequence_features = []
    for ind in int_seq_tokens:
        if ind <= 0:
            feat = np.zeros(feat_len)
        else:
            try:
                char = prot_features.chars[ind - 1]
                feat = np.array(aa_to_feat[char])
            except KeyError:
                feat = np.zeros(feat_len)
        sequence_features.append(feat)
    return np.array(sequence_features)


def write_record(my_df, record_path, y_tag, pssm_format_fi='../data/psiblast/swiss/{}_pssm.txt'):
    """
    Main function to write tensorflow records

    :param my_df: DataFrame containing sequences and labels
    :param record_path: Where you want to save file
    :param y_tag: tag of labels
    :param pssm_format_fi: path where {} corresponds to the index of the protein
    :return: 0
    """
    sequences, y, indices = my_df['sequences'].values, my_df[y_tag].astype('category'), my_df.index.values
    writer = tf.python_io.TFRecordWriter(record_path)
    for sen, label_id, id in zip(tqdm(sequences), y.cat.codes, indices):
        pssm_path = pssm_format_fi.format(id)
        try:
            pssm = pd.read_csv(pssm_path, sep=' ', skiprows=2, skipfooter=6, skipinitialspace=True)\
                .reset_index(level=[2, 3])
            pssm_feat = pssm.iloc[:MAX_LEN].values
            seq_len, n_features_pssm = pssm_feat.shape
            # print(n_features_pssm)
            pssm_mat = np.zeros(shape=(MAX_LEN, n_features_pssm))
            pssm_mat[-seq_len:] = pssm_feat
            pssm_mat = pssm_mat.reshape(-1)
            if np.isnan(pssm_mat).any():
                print('issue')

            # if seq_len != len(sen):
            #     print('Inconsistency for protein id : {}'.format(id))
            #     print(sen)
            #     print(pssm.index.values)

            tokens = [char for char in sen]
            tokens = np.array([prot_features.safe_char_to_idx(char) for char in tokens])
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
        except FileNotFoundError:
            print('{} does not exist'.format(pssm_path))
    writer.close()
    return 0


if __name__ == '__main__':
    flags.DEFINE_string('train_path', './swiss_train_data500.tfrecords', 'Path of training records')
    flags.DEFINE_string('val_path', './swiss_val_data500.tfrecords', 'Path of val records')
    FLAGS = flags.FLAGS

    train_records_filename = FLAGS.train_path
    val_records_filename = FLAGS.val_path
    # data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
    # pssm_format_file = '../../data/psiblast/riken_data/{}_pssm.txt'
    # y_name = 'is_allergenic'
    # group_name = 'species'

    data_path = '/home/pierre/riken/data/swiss/swiss_with_clans.tsv'
    pssm_format_file = '../../data/psiblast/swiss/{}_pssm.txt'
    y_name = 'clan'
    group_name = None

    df = pd.read_csv(data_path, sep='\t').dropna()
    df.loc[:, 'sequences'] = df.sequences_x

    if group_name is None:
        train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=0.2)
    else:
        df = df[df.seq_len >= 50]
        train_inds, val_inds = data_op.group_shuffle_indices(df.sequences, df[y_name], df[group_name])
        print(train_inds.shape, val_inds)
        train_df, val_df = df.iloc[train_inds], df.iloc[val_inds]

    # Writing Train data
    write_record(train_df, train_records_filename, y_tag=y_name, pssm_format_fi=pssm_format_file)
    # Writing Val data
    write_record(val_df, val_records_filename, y_tag=y_name, pssm_format_fi=pssm_format_file)

    print('Nb classes: ', len(df[y_name].unique()))
