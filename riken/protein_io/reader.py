import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

"""
    Functions to read data
"""


def get_pssm_mat(path_to_pssm, max_len):
    try:
        pssm_df = pd.read_csv(path_to_pssm, sep=' ', skiprows=2, skipfooter=6, skipinitialspace=True) \
            .reset_index(level=[2, 3])
        pssm_feat = pssm_df.iloc[:max_len].values
        seq_len, _ = pssm_feat.shape
        pssm_mat = np.zeros(shape=(max_len, 42))
        pssm_mat[-seq_len:] = pssm_feat
        if np.isnan(pssm_mat).any():
            raise ValueError
    except Exception as e:
        print(e)
        print('Error!')
        pssm_mat = np.zeros(shape=(max_len, 42))
    return pssm_mat

def read_fasta(filename):
    with open(filename) as fasta_file:  # Will close handle cleanly
        idx = []
        name = []
        description = []
        sequences = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            idx.append(seq_record.id)
            name.append(seq_record.name)
            description.append(seq_record.description)
            sequences.append(seq_record.seq)

    return pd.DataFrame({
        'idx': idx,
        'name': name,
        'description': description,
        'sequences': sequences})


def pfam_reader(fasta_path, family_clan_path):
    pfam_df = read_fasta(fasta_path)
    info = pfam_df.description.str.split(r'\s|;', expand=True) \
            .iloc[:, 0:4] \
            .rename(columns={0: 'protein_loc', 1: 'protein_tag', 2: 'family', 3: 'family_name'})
    pfam_df = pfam_df.drop(columns=['description', 'id', 'name'])
    pfam_df = pd.concat([pfam_df, info], axis=1, ignore_index=False)
    pfam_df.loc[:, 'family'] = pfam_df.family.str.split('.', expand=True).iloc[:, 0]

    family_clans = pd.read_csv(family_clan_path, sep='\t',
                               names=['family', 'clan', 'clan_name', 'family_name',
                                      'family_description'])

    pfam_df = pd.merge(left=pfam_df, right=family_clans, on='family', how='left')
    return pfam_df


def get_seqrecord(elem):
    return SeqRecord(seq=Seq(elem.sequences, IUPAC.protein),
                     id=str(elem.name), name=str(elem.name))


def offline_data_augmentation(indices_sequences, labels, switch_matrix, nb_aug=10):
    """
    Please refer to riken/riken/nn_utils/data_augmentation.py for more extensive explanation
    :param indices_sequences: list of int sequences
    :param labels:  associated labels_li
    :param switch_matrix: probability matrix used for augmentation
    :return: augmented_indices_sequences, augmented_labels
    """
    augmented_indices_sequences = []
    augmented_labels = []
    assert len(indices_sequences) == len(labels)
    cumsum_probas = switch_matrix.cumsum(axis=1)
    for sent, lbl in zip(tqdm(indices_sequences), labels):
        for _ in range(nb_aug):
            n_words = len(sent)
            dice = np.random.random(size=(n_words, 1))
            choices = (dice < cumsum_probas[sent]).argmax(axis=1)
            augmented_indices_sequences.append(choices)
            augmented_labels.append(lbl)
    return augmented_indices_sequences, augmented_labels
