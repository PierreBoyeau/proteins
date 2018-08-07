import gensim.models
from tqdm import tqdm

from riken.protein_io import reader
from riken.word2vec import amino_acid_processing

FILENAME = '../data/swiss/uniprot-reviewed%3Ayes.fasta'

if __name__ == '__main__':
    df = reader.read_fasta(FILENAME)
    print(df.info())
    all_sentences = []
    for protein_data in tqdm(df.sequences.tolist()):
        decupled_sentences = amino_acid_processing.create_words(protein_data, size=4)
        all_sentences += decupled_sentences

    w2v = gensim.models.word2vec.Word2Vec(all_sentences, sg=1, size=100, window=25,
                                          workers=16, iter=10)
    w2v.save('prot_vec_model_10_epochs_l_4.model')
