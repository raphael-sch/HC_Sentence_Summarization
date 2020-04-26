import collections
from itertools import combinations

import numpy as np
import tensorflow as tf
from utils import tf_argchoice_element


class State(
        collections.namedtuple("State", ("state", "internal_state"))):
    pass


def get_extractive_next_state_func(batch_size, sentence_length, summary_length, sentence):

    def get_next_state(state):
        boolean_map = state.internal_state
        remove_idx = tf_argchoice_element(boolean_map, element=tf.constant(True, dtype=tf.bool))
        insert_idx = tf_argchoice_element(boolean_map, element=tf.constant(False, dtype=tf.bool))

        remove_mask = tf.cast(tf.one_hot(remove_idx, sentence_length), dtype=tf.bool)
        next_boolean_map = tf.math.logical_xor(boolean_map, remove_mask)

        insert_mask = tf.cast(tf.one_hot(insert_idx, sentence_length), dtype=tf.bool)
        next_boolean_map = tf.math.logical_or(next_boolean_map, insert_mask)

        sequence = tf.broadcast_to(sentence, shape=[batch_size, sentence_length])
        flat_output = tf.boolean_mask(sequence, next_boolean_map)
        next_state = tf.reshape(flat_output, shape=(batch_size, summary_length))

        next_state = State(state=next_state, internal_state=next_boolean_map)

        return next_state

    return get_next_state


def get_extractive_initial_states(num_restarts, batch_size, x_sentence, summary_length, exhaustive=False):
    summary_length = int(summary_length)
    sentence_length = x_sentence.size

    def yield_batch(boolean_maps, states):
        initial_state = State(state=np.asarray(states), internal_state=np.asarray(boolean_maps))
        return initial_state

    boolean_maps = list()
    states = list()

    if sentence_length <= summary_length:
        states = [x_sentence]
        boolean_maps = [[True for _ in range(sentence_length)]]
    elif exhaustive:
        skipgrams = set(combinations(x_sentence, int(summary_length)))
        for skipgram in skipgrams:
            boolean_map = np.zeros(shape=sentence_length, dtype=np.bool)
            boolean_maps.append(boolean_map)
            states.append(skipgram)

            if len(states) == batch_size:
                yield yield_batch(boolean_maps, states)
                boolean_maps = list()
                states = list()
    else:
        for _ in range(num_restarts):
            boolean_map = np.zeros(shape=sentence_length, dtype=np.bool)
            idx_positive = np.random.choice(range(sentence_length), size=summary_length, replace=False)
            boolean_map[idx_positive] = True
            boolean_maps.append(boolean_map)

            state = x_sentence[np.where(boolean_map)]
            states.append(state)

            if len(states) == batch_size:
                yield yield_batch(boolean_maps, states)
                boolean_maps = list()
                states = list()
    if len(states) > 0:
        yield yield_batch(boolean_maps, states)
