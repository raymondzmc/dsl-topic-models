import numpy as np


def get_top_words(beta, vocab, num_top_words, verbose=False):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        if verbose:
            print('Topic {}: {}'.format(i, topic_str))

    return topic_str_list


def get_stopwords_set(stopwords=[]):
    if stopwords == 'English':
        from gensim.parsing.preprocessing import STOPWORDS as stopwords

    return frozenset(stopwords)


if __name__ == '__main__':
    print(list(get_stopwords_set('English'))[:10])
    print(list(get_stopwords_set())[:10])
