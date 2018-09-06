import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

"""
    Functions to read data
"""


def get_pssm_mat(path_to_pssm, max_len, padding='pre'):
    if padding not in ['pre', 'post']:
        raise ValueError('padding should either be "pre" or "post"')
    try:
        pssm_df = pd.read_csv(path_to_pssm, sep=' ', skiprows=2, skipfooter=6, skipinitialspace=True) \
            .reset_index(level=[2, 3])

        # Truncating beginning of sequences
        pssm_feat = pssm_df.iloc[-max_len:].values
        seq_len, _ = pssm_feat.shape
        pssm_mat = np.zeros(shape=(max_len, 42))

        # Applying 0-padding to begin or end based on padding option
        if padding == 'pre':
            pssm_mat[-seq_len:] = pssm_feat
        else:
            pssm_mat[:seq_len] = pssm_feat

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


def read_epitopes_data(path='/home/pierre/riken/data/riken_data/epitopes.xlsx'):
    allergenid_to_allergen_idx = (pd.read_excel(path, sheet_name='allergen2017'))
    epitopes_to_allergid = pd.read_excel(path, sheet_name='epitope2017')
    epitopes_to_seq = pd.read_excel(path, sheet_name='epitopeseq2017')

    epitopes_df = (
        pd.merge(left=epitopes_to_allergid, right=allergenid_to_allergen_idx, on='allergenid',
                 how='left')
        .merge(right=epitopes_to_seq, how='right', on='epitopeid'))

    id_cols = ['uniprot', 'original_ref', 'OtherProtACC', 'NCBI_taxID']
    epitopes_df.loc[:, id_cols] = epitopes_df[id_cols].apply(lambda x: x.astype(str).str.lower())
    return epitopes_df


def get_epitopes_masks(dataf, epitopes_dataf):
    """
    Get masks based on begin/end labels of epitopes position given in epitopes_dataf
    :param dataf:
    :param epitopes_dataf:
    :return:
    """
    def _get_masks(data_protein):
        orig_idx = data_protein.original_index.values[0]
        seq = data_protein.sequences.values[0]
        starts, ends = data_protein.start, data_protein.end
        mask = -1.0 * np.ones(shape=len(seq))

        for begin, end in zip(starts, ends):
            mask[begin:end] = 1.0
        return pd.Series({'original_index': orig_idx, 'mask': mask})

    df_cp = dataf.assign(
        trunc_index=lambda x: [elem[0] for elem in x.index.str.split('.', expand=True).values],
        original_index=lambda x: x.index.values)

    merged = pd.merge(epitopes_dataf, df_cp, left_on='uniprot', right_on='trunc_index')
    return merged.groupby('allergenid').apply(_get_masks)


def get_mask_from_epitopes_seqs(protein_seq, epitopes_ser):
    """
    Manually computes masks based on epitope SEQUENCES
    :param protein_seq:
    :param epitopes_ser:
    :return:
    """
    mask = np.zeros(len(protein_seq), dtype=np.float32)

    def _iter_routine(seq):
        starts_stops = [m.span() for m in re.finditer(pattern=seq,
                                                      string=protein_seq)]
        return starts_stops

    num_cores = multiprocessing.cpu_count()
    beg_end = Parallel(n_jobs=num_cores)(delayed(_iter_routine)(seq) for seq in epitopes_ser)
    beg_end = [tup for li in beg_end for tup in li]

    for beg, end in beg_end:
        mask[beg:end] = 1.0
    return mask


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
