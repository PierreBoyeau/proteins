import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_fasta(filename):
    with open(filename) as fasta_file:  # Will close handle cleanly
        id = []
        name = []
        description = []
        sequences = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            id.append(seq_record.id)
            name.append(seq_record.name)
            description.append(seq_record.description)
            sequences.append(seq_record.seq)

    return pd.DataFrame({
        'id': id,
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
                               names=['family', 'clan', 'clan_name', 'family_name', 'family_description'])

    pfam_df = pd.merge(left=pfam_df, right=family_clans, on='family', how='left')
    return pfam_df


def get_seqrecord(elem):
    return SeqRecord(seq=Seq(elem.sequences, IUPAC.protein),
                     id=str(elem.name), name=str(elem.name))
