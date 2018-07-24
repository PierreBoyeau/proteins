from protein_io import reader
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from prot_features import prot_features


MAX_LEN = 500
RANDOM_STATE = 42
VALUE = -1

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chars_to_idx = {char: idx+1 for (idx, char) in enumerate(chars)}


def safe_char_to_idx(char):
    if char in chars_to_idx:
        return chars_to_idx[char]
    else:
        return 0


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=value))


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
                char = chars[ind-1]
                feat = np.array(aa_to_feat[char])
            except KeyError:
                feat = np.zeros(feat_len)
        sequence_features.append(feat)
    return np.array(sequence_features)


if __name__=='__main__':
    train_tfrecords_filename = 'swiss_train_data500.tfrecords'
    val_tfrecords_filename = 'swiss_val_data500.tfrecords'
    #
    # pfam_path = '/home/pierre/riken/data/pfam/Pfam-A.fasta'
    # clans_families_path = '/home/pierre/riken/data/pfam/Pfam-A.clans.tsv'
    # df = reader.pfam_reader(pfam_path, clans_families_path).dropna()
    # df = pd.read_csv('/home/pierre/riken/data/pfam/pfam_sample.tsv', sep='\t').dropna()
    df = pd.read_csv('/home/pierre/riken/data/swiss/swiss_with_clans.tsv', sep='\t')
    df.loc[:, 'sequences'] = df.sequences_x
    train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=0.1)

    ### Writing Train data
    sequences, y = train_df['sequences'].values, train_df['clan'].astype('category')
    writer = tf.python_io.TFRecordWriter(train_tfrecords_filename)
    for sen, label_id in zip(tqdm(sequences), y.cat.codes):
        tokens = [char for char in sen]
        tokens = np.array([safe_char_to_idx(char) for char in tokens])

        padded_tokens = pad_sequences(tokens.reshape(1, -1), maxlen=MAX_LEN, value=VALUE).reshape(-1)
        padded_blosum_feat = get_feat(padded_tokens)
        feature = {
            'sentence_len': _int64_feature([len(sen)]),
            # 'sentence': _byte_feature(str.encode(sen)),
            'tokens': _int64_feature(padded_tokens),
            'blosum_feat': _float_feature(padded_blosum_feat),
            'label': _int64_feature([label_id])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

    ### Writing Val data
    sequences, y = val_df['sequences'].values, val_df['clan'].astype('category')
    writer = tf.python_io.TFRecordWriter(val_tfrecords_filename)
    for sen, label_id in zip(tqdm(sequences), y.cat.codes):
        tokens = [char for char in sen]
        tokens = np.array([safe_char_to_idx(char) for char in tokens])

        padded_tokens = pad_sequences(tokens.reshape(1, -1), maxlen=MAX_LEN).reshape(-1)
        feature = {
            'sentence_len': _int64_feature([len(sen)]),
            # 'sentence': _byte_feature(str.encode(sen)),
            'tokens': _int64_feature(padded_tokens),
            'label': _int64_feature([label_id])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

    print('Nb classes: ', len(np.unique(y)))
