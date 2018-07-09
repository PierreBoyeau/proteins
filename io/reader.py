import pandas as pd
from Bio import SeqIO


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