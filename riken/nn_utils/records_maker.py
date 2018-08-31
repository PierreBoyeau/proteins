import argparse
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from riken.protein_io import prot_features, data_op, reader

flags = tf.flags

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


def write_record(my_df, record_path, y_tag, pssm_format_fi='../data/psiblast/swiss/{}_pssm.txt',
                 max_len=None, padding='pre'):
    """
    Main function to write tensorflow records

    :param my_df: DataFrame containing sequences and labels
    :param record_path: Where you want to save file
    :param y_tag: tag of labels
    :param pssm_format_fi: path where {} corresponds to the index of the protein
    :return: 0
    """
    sequences, y, indices = my_df['sequences'].values, my_df[y_tag].values, my_df.index.values
    writer = tf.python_io.TFRecordWriter(record_path)
    for (sen, label_id, id) in zip(tqdm(sequences), y, indices):
        pssm_path = pssm_format_fi.format(id)
        try:
            tokens = [char for char in sen]
            tokens = np.array([prot_features.safe_char_to_idx(char) for char in tokens])
            tokens = pad_sequences(tokens.reshape(1, -1), maxlen=max_len,
                                   value=VALUE, padding=padding).reshape(-1)
            pssm_mat = reader.get_pssm_mat(pssm_path, max_len=max_len, padding=padding)
            pssm_mat = pssm_mat.reshape(-1)

            if np.isnan(pssm_mat).any():
                raise ValueError
            feature = {
                'sentence_len': _int64_feature([len(sen)]),
                'tokens': _int64_feature(tokens),
                'pssm_li': _float_feature(pssm_mat),
                'n_features_pssm': _int64_feature([42]),
                'label': _int64_feature([label_id])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        except FileNotFoundError:
            print('{} does not exist'.format(pssm_path))
    writer.close()
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', type=str, help='Path to train record')
    parser.add_argument('-val_path', type=str, help='Path to val record')
    parser.add_argument('-index_col', default=None, type=int, help='index_col if exists')
    parser.add_argument('-max_len', default=None, type=int,
                        help='max_len of sequences')
    parser.add_argument('-padding', default='pre', type=str, help='pre or post')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_records_filename = args.train_path
    val_records_filename = args.val_path
    index_col = args.index_col
    max_len = args.max_len
    padding = args.padding

    data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv'
    pssm_format_file = '../../data/psiblast/riken_data_v2/{}_pssm.txt'
    y_name = 'is_allergenic'
    group_name = 'predefined'

    # data_path = '/home/pierre/riken/data/riken_data/complete_from_xlsx.tsv'
    # pssm_format_file = '../../data/psiblast/riken_data/{}_pssm.txt'
    # y_name = 'is_allergenic'
    # group_name = 'species'

    # data_path = '/home/pierre/riken/data/swiss/swiss_with_clans.tsv'
    # pssm_format_file = '../../data/psiblast/swiss/{}_pssm.txt'
    # y_name = 'clan'
    # group_name = None

    df = pd.read_csv(data_path, sep='\t', index_col=index_col).dropna()
    y_ind_name = y_name+'_ind'
    label_indices, uniques = pd.factorize(df[y_name])
    print('Number of distinct classes :', len(uniques))
    df.loc[:, y_ind_name] = label_indices

    if group_name == 'predefined':
        train_df = df[df.is_train]
        val_df = df[df.is_train == False]
    elif group_name is None:
        train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=0.2)
    else:
        df = df[df.seq_len >= 50]
        train_inds, val_inds = data_op.group_shuffle_indices(df.sequences,
                                                             df[y_ind_name],
                                                             df[group_name])
        print(train_inds.shape, val_inds)
        train_df, val_df = df.iloc[train_inds], df.iloc[val_inds]

    print('{} training examples and {} test examples'.format(len(train_df), len(val_df)))

    # Writing Train data
    write_record(train_df, train_records_filename, y_ind_name, pssm_format_fi=pssm_format_file,
                 max_len=max_len, padding=padding)
    # Writing Val data
    write_record(val_df, val_records_filename, y_ind_name, pssm_format_fi=pssm_format_file,
                 max_len=max_len, padding=padding)
