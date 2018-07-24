import numpy as np
from tqdm import tqdm

data_index = 0


def create_words(seq, size=3, return_list_of_words=True):
    sentences_res = []
    for beg in range(size):
        seq_trunc = seq[beg:]
        if return_list_of_words:
            words = [seq_trunc[size*idx:size*(idx+1)] for idx in range(len(seq_trunc) // size)]
        else:
            words = ' '.join([seq_trunc[size * idx:size * (idx + 1)] for idx in range(len(seq_trunc) // size)])
        sentences_res.append(words)
    return sentences_res


def get_data_from_sentences(sentences):
    words_dict = dict()
    idx = 0
    data = []
    for sentence in tqdm(sentences):
        sentence_indices = []
        for word in sentence:
            if word in words_dict:
                idx = words_dict[word]
            else:
                idx += 1
                words_dict[word] = idx
            sentence_indices.append(idx)
        data.append(sentence_indices)
    return data, words_dict


def get_data_examples(li_proteins_code):
    all_sentences = []
    for protein_data in tqdm(li_proteins_code):
        decupled_sentences = create_words(protein_data)
        all_sentences += decupled_sentences
    data, word_dicts = get_data_from_sentences(all_sentences)
    return data, word_dicts


def get_batch_from_sentences(data, batch_size, num_skips, skip_window):
    # batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    selected_idx_stc = np.random.randint(0, len(data), size=batch_size)
    selected_sentences = data[selected_idx_stc]
    len_sentences = np.array([len(sentence) for sentence in selected_sentences])

    context_ids = np.floor(len_sentences*np.random.rand(batch_size)).astype(np.int)
    target_ids = context_ids + np.random.randint(-num_skips, num_skips+1, batch_size)
    target_ids = np.minimum(len_sentences-1, target_ids)
    target_ids = np.maximum(0, target_ids)

    batch = np.array([selected_sentences[idx][context_ids[idx]] for idx in range(batch_size)],
                     dtype=np.int32).reshape(batch_size)
    labels = np.array([selected_sentences[idx][target_ids[idx]] for idx in range(batch_size)],
                      dtype=np.int32).reshape((batch_size, 1))

    return batch, labels


if __name__ == "__main__":
    filename = '/home/pierre/riken/data/swiss/uniprot-reviewed%3Ayes.fasta'
    df = read_fasta(filename)
    # df = df[:100]
    data, dictionary = get_data_examples(df.sequences.tolist())
    data = np.array(data)

    # batch_size = 128
    # batch, labels = get_batch_from_sentences(data, batch_size, num_skips=5, skip_window=5)

    # data = [item for sentence in data for item in sentence]
    # a = generate_batch_skipgram(data, batch_size=128, num_skips=16, skip_window=8)
