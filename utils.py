import logging
import gensim
import numpy as np
import tensorflow as tf

pad_token = 'PAD'
unk_token = 'UNK'
bos_token = 'BOS'
eos_token = 'EOS'
pad_idx = 0
unk_idx = 1
bos_idx = 2
eos_idx = 3

bad_words = {'"', "'", "''", "!", "=", "-", "--", ",", "?", ".", "``", "`", "-rrb-", "-llb-", "\\/"}


def get_vocabs(vocab_file):
    word2idx = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
    for line in open(vocab_file):
        word, count = line.split()
        word2idx[word] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    assert word2idx[pad_token] == pad_idx
    assert word2idx[unk_token] == unk_idx
    assert word2idx[bos_token] == bos_idx
    assert word2idx[eos_token] == eos_idx
    return word2idx, idx2word


def get_embeddings(word2idx, s2v_file):
    if s2v_file.endswith('.npy'):
        embeddings = np.load(s2v_file)
        logging.info('loaded {} embeddings with dimension {} from npy file'.format(embeddings.shape[0],
                                                                                   embeddings.shape[1]))
    else:
        w2v_model = gensim.models.Word2Vec.load(s2v_file)
        embeddings = np.zeros(shape=(len(word2idx), w2v_model.vector_size), dtype=np.float32)
        words_not_found = list()
        for word, i in word2idx.items():
            if word in w2v_model:
                embeddings[i] = w2v_model[word]
            elif word != 'PAD':
                words_not_found.append(word)
                #logging.info('word not in embeddings: {}'.format(word))
                embeddings[i] = w2v_model['UNK']
        logging.info('{} words not in embeddings'.format(len(words_not_found)))
    return embeddings


def get_chunks(l, batch_size):
    return [l[offs:offs + batch_size] for offs in range(0, len(l), batch_size)]


def tf_argchoice_element(sequence, element):
    random_uniform = tf.random.uniform(shape=tf.shape(sequence))
    y = tf.ones(shape=tf.shape(sequence)) * -1
    random_uniform = tf.where(tf.equal(sequence, element), x=random_uniform, y=y)
    idx = tf.argmax(random_uniform, axis=-1, output_type=tf.int32)
    return idx