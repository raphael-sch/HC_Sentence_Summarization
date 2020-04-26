import logging
import tensorflow as tf

from neighbor_function import get_extractive_next_state_func
from neighbor_function import State
from lm.stitch import get_lm_metric_func
from cos.weighted_average import get_cos_metric_func


def get_model(metrics, mode='extractive'):

    if mode not in ['extractive', 'exhaustive']:
        raise ValueError('Model mode not supported: {}'.format(mode))

    inputs = dict()
    outputs = dict()

    sentence = tf.placeholder(tf.int32, shape=[None], name="sentence")
    inputs['sentence'] = sentence

    sentence_length = tf.placeholder(tf.int32, shape=[], name="sentence_length")
    inputs['sentence_length'] = sentence_length

    num_steps = tf.placeholder(tf.int32, shape=[], name="num_steps")
    inputs['num_steps'] = num_steps

    initial_state = tf.placeholder(tf.int32, shape=[None, None], name="initial_state")
    inputs['initial_state'] = initial_state

    if mode == 'extractive':
        initial_internal_state = tf.placeholder(tf.bool, shape=[None, None], name="initial_internal_state")
        inputs['initial_internal_state'] = initial_internal_state
        initial_state_tuple = State(state=initial_state, internal_state=initial_internal_state)

    summary_length = tf.placeholder(tf.int32, shape=[], name="summary_length")
    inputs['summary_length'] = summary_length

    batch_size = tf.shape(initial_state)[0]

    metric_funcs = dict()
    if 'cos' in metrics:
        logging.info('create metric function: cos')
        get_cos_metric = get_cos_metric_func(tf.expand_dims(sentence, axis=0),
                                             tf.expand_dims(sentence_length, axis=0),
                                             metrics['cos'])
        metric_funcs['cos'] = (get_cos_metric, metrics['cos']['weight'])

    if 'lm' in metrics:
        logging.info('create metric function: lm')
        get_lm_metric = get_lm_metric_func(metrics['lm'])
        metric_funcs['lm'] = (get_lm_metric, metrics['lm']['weight'])

    get_score = get_score_func(metric_funcs)

    def run_get_score():
        _states = tf.expand_dims(initial_state_tuple.state, axis=0)
        _scores = tf.expand_dims(get_score(initial_state_tuple.state), axis=0)
        return _states, _scores

    def run_get_score_exhaustive():
        _state = initial_state
        _score = get_score(initial_state)
        return _state, _score

    def run_sampler():
        next_state = get_extractive_next_state_func(batch_size, sentence_length, summary_length, sentence)

        metropolis_sampler = get_metropolis_sampler(get_score, num_steps)
        _states, _scores = metropolis_sampler(initial_state_tuple, next_state)
        return _states, _scores

    if mode == 'exhaustive':
        state, score = run_get_score_exhaustive()
        outputs['state'] = state
        outputs['score'] = score
    else:
        cond = tf.logical_or(sentence_length <= summary_length, num_steps < 1)
        states, scores = tf.cond(cond, true_fn=run_get_score, false_fn=run_sampler)
        outputs['states'] = states
        outputs['scores'] = scores

    return inputs, outputs


def get_metropolis_sampler(get_score, num_steps):
    def metropolis_sampler(initial_state, next_state):

        def body(step, state_old, score_old, states_ta, scores_ta):
            state_new = next_state(state_old)
            score_new = get_score(state_new.state)

            states_ta = states_ta.write(step, state_new.state)
            scores_ta = scores_ta.write(step, score_new)

            cond = tf.greater_equal(score_new, score_old)

            state_next = tf.where(cond, x=state_new.state, y=state_old.state)
            internal_state = tf.where(cond, x=state_new.internal_state, y=state_old.internal_state)
            score_next = tf.where(cond, x=score_new, y=score_old)

            state_next = State(state=state_next, internal_state=internal_state)
            step += 1
            return step, state_next, score_next, states_ta, scores_ta

        def condition(step, unused_state, unused_score, unused_states_ta, unused_scores_ta):
            return step < num_steps

        initial_step = 0
        initial_score = get_score(initial_state.state)

        initial_states_ta = tf.TensorArray(dtype=initial_state.state.dtype,
                                           size=num_steps,
                                           element_shape=initial_state.state.shape,
                                           dynamic_size=False)
        initial_scores_ta = tf.TensorArray(dtype=initial_score.dtype,
                                           size=num_steps,
                                           element_shape=initial_score.shape,
                                           dynamic_size=False)

        initial_states_ta = initial_states_ta.write(initial_step, initial_state.state)
        initial_scores_ta = initial_scores_ta.write(initial_step, initial_score)
        initial_step += 1

        res = tf.while_loop(cond=condition,
                            body=body,
                            loop_vars=[initial_step,
                                       initial_state,
                                       initial_score,
                                       initial_states_ta,
                                       initial_scores_ta],
                            back_prop=False)
        states = res[3].stack()
        scores = res[4].stack()

        return states, scores
    return metropolis_sampler


def get_score_func(metric_funcs):
    if len(metric_funcs) == 0:
        return 'Need metrics to compute score'

    def get_score(sample, sample_length=None):
        if sample_length is None:
            sample_length = tf.reduce_sum(tf.ones(shape=(tf.shape(sample)), dtype=tf.int32), axis=-1)
        score = 1

        for name, (metric_func, weight) in metric_funcs.items():
            logging.info(f'score name: {name}')
            logging.info(f'score weight: {weight}')
            score_metric = metric_func(sample, sample_length)
            score = score * tf.pow(score_metric, weight)

        return score
    return get_score
