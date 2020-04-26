import numpy as np
import logging


def main():
    input_file = 'train.txt'
    output_file = 'outputs/article.idf'


    document_freq = dict()
    N = 0

    for line in open(input_file):
        N += 1
        words = line.split()
        for word in set(words):
            if word in document_freq:
                document_freq[word] += 1
            else:
                document_freq[word] = 1

    with open(output_file, 'w') as f:
        for word, df in document_freq.items():
            idf = np.log2(np.divide(N, df))
            f.write('{} {}\n'.format(word, str(idf)))


def get_idf_vector(idf_file, word2idx):
    words_not_found = list()
    vector = np.zeros(shape=(len(word2idx)), dtype=np.float32)
    for line in open(idf_file):
        word, idf = line.split()
        idf = float(idf)
        if word in word2idx:
            idx = word2idx[word]
            vector[idx] = idf
        else:
            words_not_found.append(word)
    logging.info('{} words not in idf'.format(len(words_not_found)))
    return vector


if __name__ == '__main__':
    main()
