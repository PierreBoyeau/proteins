import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from riken.prot_features import prot_features

flags = tf.flags

flags.DEFINE_string('train_path', './swiss_train_data500.tfrecords', 'Path of training records')
flags.DEFINE_string('val_path', './swiss_val_data500.tfrecords', 'Path of val records')
FLAGS = flags.FLAGS


MAX_LEN = 500
RANDOM_STATE = 42
VALUE = -1

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chars_to_idx = {char: idx+1 for (idx, char) in enumerate(chars)}
blosom_80 = prot_features.get_blosum80_dict_to_features()
# blosom_80['U'] = np.zeros()


def safe_char_to_idx(char):
    if char in chars_to_idx:
        return chars_to_idx[char]
    else:
        return


def create_overall_static_aa_mat():
    res_mat = np.concatenate([create_blosom_80_mat(), create_amino_acids_prop_mat()], axis=1)
    return res_mat


def create_blosom_80_mat():
    len_mat = len(blosom_80['A'])
    zeros = np.zeros(len_mat)
    mat = [zeros]  # Value for 0 index
    for char in chars:
        if char in blosom_80:
            mat.append(blosom_80[char])
        else:
            mat.append(zeros)
    return np.array(mat)


def create_amino_acids_prop_mat():
    prop_df = prot_features.get_amino_acids_chemical_properties()
    prop_df = prop_df.reindex(['NULL']+chars).fillna(0)
    return prop_df.values


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
                char = chars[ind-1]
                feat = np.array(aa_to_feat[char])
            except KeyError:
                feat = np.zeros(feat_len)
        sequence_features.append(feat)
    return np.array(sequence_features)


def write_record(my_df, record_path, pssm_format_file='../data/psiblast/swiss/{}_pssm.txt'):
    sequences, y, indices = my_df['sequences'].values, my_df['clan'].astype('category'), my_df.index.values
    writer = tf.python_io.TFRecordWriter(record_path)
    for sen, label_id, id in zip(tqdm(sequences), y.cat.codes, indices):
        pssm_path = pssm_format_file.format(id)
        pssm = pd.read_csv(pssm_path, sep=' ', skiprows=2, skipfooter=6, skipinitialspace=True)\
            .reset_index(level=[2, 3])
        pssm_feat = pssm.iloc[:MAX_LEN].values
        seq_len, n_features_pssm = pssm_feat.shape
        pssm_mat = np.zeros(shape=(MAX_LEN, n_features_pssm))
        pssm_mat[-seq_len:] = pssm_feat
        pssm_mat = pssm_mat.reshape(-1)

        if seq_len != len(sen):
            print('Inconsistency for protein id : {}'.format(id))
            print(sen)
            print(pssm.index.values)

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
    train_records_filename = FLAGS.train_path
    val_records_filename = FLAGS.val_path

    # pfam_path = '/home/pierre/riken/data/pfam/Pfam-A.fasta'
    # clans_families_path = '/home/pierre/riken/data/pfam/Pfam-A.clans.tsv'
    # df = reader.pfam_reader(pfam_path, clans_families_path).dropna()

    df = pd.read_csv('/home/pierre/riken/data/swiss/swiss_with_clans.tsv', sep='\t')
    df.loc[:, 'sequences'] = df.sequences_x
    train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=0.1)

    # Writing Train data
    write_record(train_df, train_records_filename)
    # Writing Val data
    write_record(val_df, val_records_filename)

    print('Nb classes: ', len(np.unique(y)))
