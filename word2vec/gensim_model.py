import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../io'))

import reader
import amino_acid_processing
import gensim.models
from tqdm import tqdm

if __name__ == '__main__':
    filename = '../data/swiss/uniprot-reviewed%3Ayes.fasta'
    df = reader.read_fasta(filename)
    df = df[:1000]
    print(df.info())
    all_sentences = []
    for protein_data in tqdm(df.sequences.tolist()):
        decupled_sentences = amino_acid_processing.create_words(protein_data)
        all_sentences += decupled_sentences

    w2v = gensim.models.word2vec.Word2Vec(all_sentences, sg=1, size=100, window=25, workers=16)
    w2v.save('prot_vec_model.model')