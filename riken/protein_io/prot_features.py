import os
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SubsMat import MatrixInfo
import numpy as np
import pandas as pd

HYDROPATHS = {
    'A': 1.800,
    'R': -4.500,
    'N': -3.500,
    'D': -3.500,
    'C': 2.500,
    'Q': -3.500,
    'E': -3.500,
    'G': -0.400,
    'H': -3.200,
    'I': 4.500,
    'L': 3.800,
    'K': -3.900,
    'M': 1.900,
    'F': 2.800,
    'P': -1.600,
    'S': -0.800,
    'T': -0.700,
    'W': -0.900,
    'Y': -1.300,
    'V': 4.200,
}

let1_to_let3 = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'E': 'Glu', 'Q': 'Gln', 'G': 'Gly',
                'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser',
                'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', }

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chars_to_idx = {char: idx + 1 for (idx, char) in enumerate(chars)}

aa_frequencies = {"Ala": 8.25, "Arg": 5.53, "Asn": 4.06, "Asp": 5.45, "Cys": 1.37, "Gln": 3.93, "Glu": 6.75,
                  "Gly": 7.07, "His": 2.27, "Ile": 5.96, "Leu": 9.66, "Lys": 5.84, "Met": 2.42, "Phe": 3.86,
                  "Pro": 4.70, "Ser": 6.56, "Thr": 5.34, "Trp": 1.08, "Tyr": 2.92, "Val": 6.87, }


def create_overall_static_aa_mat(normalize=True):
    res_mat = np.concatenate([create_blosom_80_mat(), create_amino_acids_prop_mat()], axis=1)
    if normalize:
        res_mat = (res_mat - res_mat.mean(axis=0)) / res_mat.std(axis=0)
    return res_mat


def create_blosom_80_mat():
    blosom_80 = get_blosum80_dict_to_features()
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
    prop_df = get_amino_acids_chemical_properties()
    prop_df = prop_df.reindex(['NULL'] + chars).fillna(0)
    return prop_df.values


def create_kidera_mat():
    kidera_dict = get_kidera_dict_to_features()
    zeros = np.zeros(10)
    mat = [zeros]  # Value for 0 index

    for char in chars:
        if char in kidera_dict:
            mat.append(kidera_dict[char])
        else:
            mat.append(zeros)
    res = np.array(mat)
    assert res.shape[1] == 10
    return res
            

def get_amino_acids_chemical_properties():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_csv = os.path.join(dir_path, '../../data/amino_acid_properties.tsv')

    properties_df = pd.read_csv(path_to_csv, sep='\t').drop(labels="amino acid", axis=1).set_index(keys='character')
    feature_means = properties_df.mean(axis=0)
    properties_df = properties_df.fillna(feature_means)
    properties_df = (properties_df - feature_means) / properties_df.std(axis=0)
    return properties_df


def get_blosum80_dict_to_features():
    """
    Used in 1503.01919
    :return:
    """
    partial_blosum_mat = MatrixInfo.blosum80
    blosum_mat = dict()
    for (aa1, aa2) in partial_blosum_mat:
        blosum_mat[(aa1, aa2)] = partial_blosum_mat[(aa1, aa2)]
        blosum_mat[(aa2, aa1)] = partial_blosum_mat[(aa1, aa2)]

    keys = blosum_mat.keys()
    keys = np.unique([aa1 for (aa1, aa2) in keys])
    aa_to_blosum_features = dict()
    for aa in keys:
        aa_to_blosum_features[aa] = np.array([blosum_mat[aa, other_aa] for other_aa in keys])
    return aa_to_blosum_features


def get_kidera_dict_to_features():
    kidera_features = {'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.20, -0.08, 0.21, -0.48],
                       'B': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.10],
                       'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.70],
                       'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.40, 0.04, 0.38, -0.35, -0.12],
                       'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.10, 1.71, -0.44],
                       'G': [1.46, -1.96, -0.23, -0.16, 0.10, -0.11, 1.32, 2.36, -1.66, 0.46],
                       'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
                       'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
                       'K': [-0.34, 0.82, -0.23, 1.70, 1.54, -1.62, 1.15, -0.08, -0.48, 0.60],
                       'L': [-1.04, 0.00, -0.24, -1.10, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
                       'M': [-1.40, 0.18, -0.42, -0.73, 2.00, 1.52, 0.26, 0.11, -1.27, 0.27],
                       'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.10, -1.73],
                       'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.30, -2.30, 0.74, -0.28],
                       'Q': [-0.47, 0.24, 0.07, 1.10, 1.10, 0.59, 0.84, -0.71, -0.03, -2.33],
                       'R': [0.22, 1.27, 1.37, 1.87, -1.70, 0.46, 0.92, -0.39, 0.23, 0.93],
                       'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
                       'T': [0.26, -0.70, 1.21, 0.63, -0.10, 0.21, 0.24, -1.15, -0.56, 0.19],
                       'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'V': [-0.74, -0.71, 2.04, -0.40, 0.50, -0.81, -1.07, 0.06, -0.46, 0.65],
                       'W': [0.30, 2.10, -0.72, -1.57, -1.16, 0.57, -0.48, -0.40, -2.30, -0.60],
                       'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'Y': [1.38, 1.48, 0.80, -0.56, 0.00, -0.68, -0.31, 1.03, -0.05, 0.53],
                       'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    return kidera_features


def molecular_weight(seq):
    return ProteinAnalysis(filter_seq(seq)).molecular_weight()


def aromaticity(seq):
    return ProteinAnalysis(filter_seq(seq)).aromaticity()


def isoelectric_point(seq):
    return ProteinAnalysis(filter_seq(seq)).isoelectric_point()


def instability_index(seq):
    return ProteinAnalysis(filter_seq(seq)).instability_index()


def aliphatic_index(seq):
    _a = 2.9
    _b = 3.9

    L = len_proteins(seq)
    x_ala = len([char for char in seq if char == 'A']) / L
    x_val = len([char for char in seq if char == 'V']) / L
    x_ile_leu = len([char for char in seq if char in ['I', 'L']]) / L

    return x_ala + _a * x_val + _b * x_ile_leu


def gravy(seq):
    sum_hydropath = np.array([HYDROPATHS[aa] for aa in filter_seq(seq)]).sum()
    return sum_hydropath / len_proteins(seq)


def len_proteins(seq):
    return len(seq)


def filter_seq(seq):
    return ''.join([aa for aa in seq if aa in HYDROPATHS])


if __name__ == '__main__':
    seq = 'MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGT'
    print(molecular_weight(seq))
    print(aromaticity(seq))
    print(instability_index(seq))
    print(aliphatic_index(seq))
    print(gravy(seq))
    print(len_proteins(seq))

    analysis = ProteinAnalysis(seq)
