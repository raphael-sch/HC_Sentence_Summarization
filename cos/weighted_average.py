import logging
import tensorflow as tf


def get_cos_metric_func(sentence, sentence_length, cos_utils):

    def _cosine_similarity(sample_emb):
        sample_emb = tf.nn.l2_normalize(sample_emb, axis=-1)
        sim = tf.reduce_sum(tf.multiply(sample_emb, sentence_emb), axis=-1)
        return sim

    sentence_weights = None
    if 'idf' in cos_utils:
        logging.info('apply tfidf')
        sentence_weights = get_weights(sentence, sentence_length, cos_utils['idf'])
    sentence_emb = get_emb_weighted_average(sentence,
                                            sentence_length,
                                            cos_utils['embeddings'],
                                            weights=sentence_weights)
    sentence_emb = tf.nn.l2_normalize(sentence_emb, axis=-1)

    def get_cos_metric(sample, sample_length):
        sample_weights = None
        if 'idf' in cos_utils:
            logging.info('apply tfidf')
            sample_weights = get_weights(sample, sample_length, cos_utils['idf'])
        sample_emb = get_emb_weighted_average(sample,
                                              sample_length,
                                              cos_utils['embeddings'],
                                              weights=sample_weights)
        cos_sim = _cosine_similarity(sample_emb)
        score = (cos_sim + 1) / 2
        return score

    return get_cos_metric


def get_weights(sequence, sequence_length, idf_vector):
    weights = tf.gather(idf_vector, sequence)
    return weights


def get_emb_weighted_average(sequence, seq_length, embeddings, weights=None):
    if weights is None:
        weights = tf.ones_like(sequence, dtype=tf.float32)
    mask = tf.sequence_mask(seq_length, dtype=tf.float32)
    weights = weights * mask
    weights = tf.expand_dims(weights, axis=-1)
    emb_seq = tf.nn.embedding_lookup(embeddings, sequence)
    return tf.reduce_sum(weights * emb_seq, axis=1) / tf.expand_dims(tf.cast(seq_length, dtype=tf.float32), axis=-1)
