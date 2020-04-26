import os
import tensorflow as tf
from tensorflow.python.training.checkpoint_utils import init_from_checkpoint
from utils import bos_idx, pad_idx, eos_idx
from lm.language_model import get_model, get_config


def get_lm_metric_func(lm_params):
    def get_lm_metric(sample, sample_length):
        return get_language_model_score(sample, sample_length, lm_params)
    return get_lm_metric


def sample_to_language_model_input(sample, sample_length):
    batch_size = tf.shape(sample)[0]
    pad_idxs = tf.tile([[pad_idx]], [batch_size, 1])
    max_sample_length = tf.reduce_max(sample_length)
    y_mask = tf.one_hot(sample_length, depth=max_sample_length + 1, on_value=True, off_value=False)

    bos_idxs = tf.tile([[bos_idx]], [batch_size, 1])
    x_lm = tf.concat((bos_idxs, sample), axis=-1)
    y_lm = tf.concat((sample, pad_idxs), axis=-1)
    eos_idxs = tf.ones_like(y_lm) * eos_idx
    y_lm = tf.where(y_mask, eos_idxs, y_lm)

    seq_length = sample_length + 1
    return dict(x=x_lm, y=y_lm, seq_length=seq_length)


def get_reverse_inputs(stitch_inputs):
    x_lm = stitch_inputs['x']
    y_lm = stitch_inputs['y']
    seq_length = stitch_inputs['seq_length']
    x_rev_lm = tf.reverse_sequence(y_lm, seq_length, seq_axis=1, batch_axis=0)
    y_rev_lm = tf.reverse_sequence(x_lm, seq_length, seq_axis=1, batch_axis=0)
    return dict(x=x_rev_lm, y=y_rev_lm, seq_length=seq_length)


def get_language_model_score(sample, sample_length, lm_params):
    stitch_inputs = sample_to_language_model_input(sample, sample_length)

    with tf.variable_scope('LanguageModel', reuse=tf.AUTO_REUSE):
        config_forward = get_config(lm_params['forward'])
        config_forward['sampled_softmax'] = 0
        _, outputs = get_model(config_forward,
                               num_words=lm_params['num_words'],
                               stitch_inputs=stitch_inputs)
    losses = outputs['losses']

    print(lm_params['reverse'])
    if lm_params['reverse'] is not None:
        stitch_inputs_rev = get_reverse_inputs(stitch_inputs)
        with tf.variable_scope('LanguageModelReverse', reuse=tf.AUTO_REUSE):
            config_reverse = get_config(lm_params['reverse'])
            config_reverse['sampled_softmax'] = 0
            _, outputs_rev = get_model(config_reverse,
                                       num_words=lm_params['num_words'],
                                       stitch_inputs=stitch_inputs_rev)
        losses_rev = outputs_rev['losses']
        losses = (losses + losses_rev) / 2

    perplexities = tf.exp(losses)
    #perplexities = tf.Print(perplexities, [perplexities], summarize=500)
    return 10000 / perplexities


def init_lm_checkpoints(lm_dirs):
    assignment_map = {'LanguageModel/': 'LanguageModel/'}
    init_from_checkpoint(os.path.join(lm_dirs['forward'], 'ckpt'), assignment_map=assignment_map)
    if lm_dirs['reverse'] is not None:
        assignment_map = {'LanguageModel/': 'LanguageModelReverse/'}
        init_from_checkpoint(os.path.join(lm_dirs['reverse'], 'ckpt'), assignment_map=assignment_map)
