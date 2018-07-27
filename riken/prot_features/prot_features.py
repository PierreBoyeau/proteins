import os
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SubsMat import MatrixInfo
import numpy as np
import pandas as pd

HYDROPATHS = {
    'A':  1.800,
    'R': -4.500,
    'N': -3.500,
    'D': -3.500,
    'C':  2.500,
    'Q': -3.500,
    'E': -3.500,
    'G': -0.400,
    'H': -3.200,
    'I':  4.500,
    'L':  3.800,
    'K': -3.900,
    'M':  1.900,
    'F':  2.800,
    'P': -1.600,
    'S': -0.800,
    'T': -0.700,
    'W': -0.900,
    'Y': -1.300,
    'V':  4.200,
}


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
    x_ala = len([char for char in seq if char=='A']) / L
    x_val = len([char for char in seq if char=='V']) / L
    x_ile_leu = len([char for char in seq if char in ['I', 'L']]) / L

    return x_ala + _a*x_val + _b*x_ile_leu


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
