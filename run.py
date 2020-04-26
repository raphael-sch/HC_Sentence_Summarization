import argparse
import os
import logging
import sys
import time
import gzip
import pickle
from shutil import copy
from math import factorial

import yaml
import numpy as np
import tensorflow as tf

from lm.stitch import init_lm_checkpoints
from cos.tfidf import get_idf_vector
from model import get_model
from utils import get_vocabs, get_embeddings, bad_words, unk_idx
from neighbor_function import get_extractive_initial_states

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='test/input.txt')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/default')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--from_line', type=int, default=0)
    parser.add_argument('--to_line', type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log_{}_{}.txt'.format(args.from_line, args.to_line))
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)
    if args.to_line == -1:
        args.to_line = sum(1 for _ in open(args.input_file))
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))
    config = yaml.safe_load(open(args.config))
    copy(args.config, os.path.join(args.output_dir, 'config.yaml'))
    copy(config['vocab_file'], os.path.join(args.output_dir, 'vocab.txt'))

    if config['mode'] not in ['extractive', 'extractive_norm']:
        raise ValueError('mode not supported: {}'.format(config['mode']))
    run(args, config)


def run(args, config):
    time_start = time.time()
    print('time_start: {}'.format(str(time_start)))

    sess, model_inputs, model_outputs, embeddings, word2idx, idx2word = build_graph(config)

    lines = get_lines(args)
    all_max_scores = list()
    for line_id, sentence in lines.items():
        logging.info('line id: {}'.format(line_id))
        line_output_dir = os.path.join(args.output_dir, 'lines', str(line_id))
        os.makedirs(line_output_dir, exist_ok=True)

        all_outputs = list()
        line_time_start = time.time()
        for batch_id, batch in enumerate(get_batches(sentence, line_id, args.batch_size, args, config, word2idx)):
            x_sentence, sentence_length, initial_state, summary_length, num_steps = batch
            logging.info('batch id: {}'.format(batch_id))
            print(x_sentence)
            feed_dict = {
                model_inputs['sentence']: x_sentence,
                model_inputs['sentence_length']: sentence_length,
                model_inputs['initial_state']: initial_state.state,
                model_inputs['initial_internal_state']: initial_state.internal_state,
                model_inputs['summary_length']: summary_length,
                model_inputs['num_steps']: num_steps
                         }

            outputs = sess.run(model_outputs, feed_dict=feed_dict)
            all_outputs.append(outputs)

        states = np.concatenate([o['states'] for o in all_outputs], axis=1)
        scores = np.concatenate([o['scores'] for o in all_outputs], axis=1)

        max_idx = np.unravel_index(np.argmax(scores), scores.shape)
        max_score = scores[max_idx]
        all_max_scores.append(max_score)
        max_state = states[max_idx]
        logging.info('max_score: {}'.format(max_score))
        logging.info('max_state: {}'.format(max_state))
        logging.info('summary: {}'.format(' '.join([idx2word[idx] for idx in max_state])))
        logging.info('run_time: {}'.format(int(time.time()-line_time_start)))

        outputs = dict(states=states,
                       scores=scores,
                       sentence=sentence,
                       line_id=line_id,
                       time=time.time()-line_time_start)
        with gzip.open(os.path.join(line_output_dir, 'outputs.pickle.gzip'), 'wb') as f:
            pickle.dump(outputs, f)
    logging.info('')
    logging.info('avg max score: {}'.format(np.mean(all_max_scores)))
    logging.info('end: {}'.format(str(time.time())))
    logging.info('runtime: {}'.format(str(time.time() - time_start)))


def build_graph(config):

    word2idx, idx2word = get_vocabs(config['vocab_file'])
    embeddings = get_embeddings(word2idx, config['s2v_file'])

    weights = config.get('weights', [1 for _ in config['metrics']])
    assert len(config['metrics']) == len(weights)
    metrics = {m: {'weight': w} for m, w in zip(config['metrics'], weights)}

    if 'lm' in metrics:
        metrics['lm'].update(dict(forward=config['lm_save_dir'],
                                  reverse=config.get('lm_rev_save_dir', None),
                                  num_words=len(word2idx)))

    if 'cos' in metrics:
        idf_file = config.get('idf_file', None)
        if idf_file is not None:
            metrics['cos'].update(dict(idf=get_idf_vector(idf_file, word2idx), embeddings=embeddings))
        else:
            metrics['cos'].update(dict(embeddings=embeddings))

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=sess_config)

    model_inputs, model_outputs = get_model(metrics, mode=config['mode'])

    if 'lm' in metrics:
        init_lm_checkpoints(metrics['lm'])
    sess.run(tf.global_variables_initializer())
    return sess, model_inputs, model_outputs, embeddings, word2idx, idx2word


def get_lines(args):
    lines = dict()
    for line_id, line in enumerate(open(args.input_file)):
        if line_id in range(args.from_line, args.to_line):
            sentence = line.split()
            lines[line_id] = sentence
    return lines


def get_summary_length(config, sentence_length):
    summary_length_target = config.get('summary_length', 8)
    if type(summary_length_target) == str and summary_length_target.endswith('p'):
        summary_percent = int(config['summary_length'][:-1]) / 100.0
        summary_length_target = max(1, int(round(sentence_length * summary_percent)))
    return np.asarray(summary_length_target, dtype=np.int32)


def get_batches(sentence, line_id, batch_size, args, config, word2idx):
    orig_sentence_length = len(sentence)
    sentence = [w for w in sentence if w not in bad_words]

    x_sentence = np.asarray([word2idx.get(w, unk_idx) for w in sentence], dtype=np.int32)
    sentence_length = np.asarray(len(sentence), dtype=np.int32)
    summary_length = get_summary_length(config, orig_sentence_length)
    logging.info('sentence: {}'.format(' '.join(sentence)))
    logging.info('sentence_length: {}'.format(sentence_length))
    logging.info('summary_length: {}'.format(summary_length))

    num_steps = max(1, int((sentence_length * summary_length**2) * config.get('steps_factor', 0.1)))
    num_restarts = max(1, int((sentence_length * summary_length**2) * config.get('restarts_factor', 0.01)))

    #num_steps = min(800, num_steps)
    #num_restarts = min(300, num_restarts)

    #batch_size = int(batch_size * 11 / summary_length)

    num_evaluations = num_steps * num_restarts

    logging.info('number of evaluations: {}'.format(num_evaluations))
    logging.info('number of restarts: {}'.format(num_restarts))
    logging.info('number of steps: {}'.format(num_steps))

    if config['mode'] == 'extractive':
        num_exhaustive = int(factorial(sentence_length) / factorial(summary_length) / factorial(max(1, sentence_length - summary_length)))
        exhaustive = num_evaluations > num_exhaustive and config.get('allow_exhaustive', False)
        if exhaustive:
            logging.info('roughly number of exhaustive evaluations: {}'.format(num_exhaustive))
        for initial_state in get_extractive_initial_states(num_restarts,
                                                           batch_size,
                                                           x_sentence,
                                                           summary_length,
                                                           exhaustive=exhaustive):
            if exhaustive:
                num_steps = 0

            yield x_sentence, sentence_length, initial_state, summary_length, num_steps


if __name__ == '__main__':
    main()
