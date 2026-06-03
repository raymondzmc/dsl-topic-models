import numpy as np
import gensim.downloader
import scipy.sparse
from tqdm import tqdm


def get_word_embeddings(vocab, embedding_model='glove-wiki-gigaword-200'):
    glove_vectors = gensim.downloader.load(embedding_model)
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

    num_found = 0
    for i, word in enumerate(tqdm(vocab, desc="===>Creating word embeddings")):
        try:
            key_word_list = glove_vectors.index_to_key
        except:
            key_word_list = glove_vectors.index2word

        if word in key_word_list:
            word_embeddings[i] = glove_vectors[word]
            num_found += 1

    print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(word_embeddings)
