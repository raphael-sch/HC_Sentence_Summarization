import tensorflow as tf
import logging
import numpy as np


def get_chunks(l, batch_size):
    return [l[offs:offs + batch_size] for offs in range(0, len(l), batch_size)]


def tf_argchoice_element(sequence, element):
    random_uniform = tf.random.uniform(shape=tf.shape(sequence))
    y = tf.ones(shape=tf.shape(sequence)) * -1
    random_uniform = tf.where(tf.equal(sequence, element), x=random_uniform, y=y)
    idx = tf.argmax(random_uniform, axis=-1, output_type=tf.int32)
    return idx


def get_embeddings(s2v_file):
    embeddings = np.load(s2v_file)
    logging.info('loaded {} embeddings with dimension {} from npy file'.format(embeddings.shape[0],
                                                                               embeddings.shape[1]))
    return embeddings
