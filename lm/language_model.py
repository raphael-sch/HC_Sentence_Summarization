import argparse
import time
import os
import sys
from shutil import copyfile
from random import shuffle
import logging

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.array_ops import sequence_mask
import numpy as np
import yaml

from utils import get_vocabs, get_embeddings, unk_idx, pad_idx, bos_idx, eos_idx, bad_words

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='train/train.title.txt')
    parser.add_argument('--valid_file', type=str, default='train/eval/references/title.txt')
    parser.add_argument('--vocab_file', type=str, default='train/vocab_title.txt')
    parser.add_argument('--w2v_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config_file', type=str, default='configs/default.yaml')
    parser.add_argument('--save_dir', type=str, default='outputs/default')
    args = parser.parse_args()

    logging.info(str(args))

    if not os.path.exists(args.save_dir) and args.mode == 'test':
        raise ValueError('save dir not found')

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'train':
        file_handler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
        logging.getLogger().addHandler(file_handler)
        logging.info(str(args))
        logging.info('start training')

        config = yaml.safe_load(open(args.config_file))
        word2idx, model_inputs, model_outputs, sess = prepare(args, config)
        yaml.dump(config, open(os.path.join(args.save_dir, 'config.yaml'), 'w'))
        copyfile(args.vocab_file, os.path.join(args.save_dir, 'vocab.txt'))
        for name, value in config.items():
            logging.info('{}: {}'.format(name, value))
        train(args, config, word2idx, model_inputs, model_outputs, sess)
    else:
        file_handler = logging.FileHandler(os.path.join(args.save_dir, 'log_valid.txt'))
        logging.getLogger().addHandler(file_handler)
        logging.info(str(args))
        logging.info('start validation')

        config_file = os.path.join(args.save_dir, 'config_valid.yaml')
        if not os.path.isfile(config_file):
            config_file = os.path.join(args.save_dir, 'config.yaml')
        config = yaml.load(open(config_file))
        config['sampled_softmax'] = 0
        args.vocab_file = os.path.join(args.save_dir, 'vocab.txt')
        word2idx, model_inputs, model_outputs, sess = prepare(args, config)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(os.path.join(args.save_dir, 'ckpt'))
        saver.restore(sess, ckpt)

        for name, value in config.items():
            logging.info('{}: {}'.format(name, value))
        valid(args, config, word2idx, model_inputs, model_outputs, sess)


def get_config(save_dir, file_name='config.yaml'):
    config_file = os.path.join(save_dir, file_name)
    return yaml.load(open(config_file))


def prepare(args, config):
    word2idx, idx2word = get_vocabs(args.vocab_file)
    try:
        embeddings = get_embeddings(word2idx, args.w2v_file)
    except FileNotFoundError:
        logging.info('embedding file not found. Train embeddings from scratch instead')
        embeddings = None
    with tf.variable_scope('LanguageModel'):
        model_inputs, model_outputs = get_model(config, embeddings, len(word2idx))

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    return word2idx, model_inputs, model_outputs, sess


def get_model(config, embeddings=None, num_words=None, stitch_inputs=None):
    inputs = dict()
    outputs = dict()

    if stitch_inputs is None:
        inputs['x'] = tf.placeholder(tf.int32, shape=[None, None], name="x")
        inputs['y'] = tf.placeholder(tf.int32, shape=[None, None], name="y")
        inputs['seq_length'] = tf.placeholder(tf.int32, shape=[None], name="seq_length")
    else:
        inputs['x'] = stitch_inputs['x']
        inputs['y'] = stitch_inputs['y']
        inputs['seq_length'] = stitch_inputs['seq_length']

    if embeddings is None:
        logging.info('initialize embeddings')
        embeddings = tf.get_variable(name="embedding",
                                     shape=[num_words, config['embedding_size']],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.1),
                                     trainable=True)
    else:
        logging.info('use pretrained embeddings')
        logging.info('embeddings trainable: {}'.format(config.get('embedding_trainable', False)))
        embeddings = tf.get_variable("embeddings",
                                     shape=embeddings.shape,
                                     initializer=tf.constant_initializer(embeddings),
                                     trainable=config.get('embedding_trainable', False))

    inputs['input_keep_prob'] = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32),
                                                            shape=[],
                                                            name="input_keep_prob")
    inputs['output_keep_prob'] = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32),
                                                             shape=[],
                                                             name="output_keep_prob")
    inputs['learning_rate'] = tf.placeholder_with_default(tf.constant(config['learning_rate'], dtype=tf.float32),
                                                          shape=[],
                                                          name="learning_rate")
    batch_size = tf.shape(inputs['x'])[0]

    def create_cell():
        rnn_cell_type = config.get('rnn_cell', 'lnlstm')
        if rnn_cell_type == 'lstm':
            logging.info('Use LSTMBlockCell cell')
            _cell = tf.contrib.rnn.LSTMBlockCell(config['rnn_size'])
        else:
            logging.info('Use LayerNormBasicLSTMCell cell')
            _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(config['rnn_size'])
        _cell = tf.nn.rnn_cell.DropoutWrapper(_cell,
                                              input_keep_prob=inputs['input_keep_prob'],
                                              output_keep_prob=inputs['output_keep_prob'])
        return _cell

    cells = [create_cell() for _ in range(config['num_layers'])]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    x_embedded = tf.nn.embedding_lookup(embeddings, inputs['x'])
    helper = seq2seq.TrainingHelper(x_embedded, inputs['seq_length'])

    projection_layer = Dense(embeddings.shape[0], name='projection_layer', use_bias=True, dtype=tf.float32)
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    mask = sequence_mask(inputs['seq_length'], dtype=tf.float32)

    decoder = seq2seq.BasicDecoder(cell,
                                   helper,
                                   initial_state=initial_state)
    decode_output, _, _ = seq2seq.dynamic_decode(decoder,
                                                 impute_finished=True,
                                                 swap_memory=config.get('swap_memory', False))

    if config.get('sampled_softmax', 0) > 0:
        projection_layer.build(input_shape=decode_output.rnn_output.shape)

        def _sampled_loss(labels, logits):
            return tf.nn.sampled_softmax_loss(tf.transpose(projection_layer.kernel),
                                              projection_layer.bias,
                                              tf.expand_dims(labels, -1),
                                              logits,
                                              num_sampled=config['sampled_softmax'],
                                              num_classes=num_words)

        softmax_loss_function = _sampled_loss
        logits_input = decode_output.rnn_output
    else:
        softmax_loss_function = None
        logits_input = projection_layer(decode_output.rnn_output)

    losses = seq2seq.sequence_loss(logits_input,
                                   inputs['y'],
                                   mask,
                                   softmax_loss_function=softmax_loss_function,
                                   average_across_batch=False,
                                   average_across_timesteps=False)

    outputs['total_loss'] = tf.reduce_sum(losses)
    outputs['num_tokens'] = tf.reduce_sum(mask)

    outputs['loss'] = outputs['total_loss'] / outputs['num_tokens']
    outputs['perplexity'] = tf.exp(outputs['loss'], name='perplexity')

    if stitch_inputs is not None:
        losses = seq2seq.sequence_loss(logits_input, inputs['y'], mask, average_across_batch=False)
        outputs['losses'] = tf.identity(losses, name='losses')
        outputs['perplexities'] = tf.exp(losses, name='perplexities')

    if stitch_inputs is None:
        with tf.variable_scope('Optimizer'):
            optimizer_name = config.get('optimizer', 'sgd')
            if optimizer_name == 'adam':
                logging.info('use adam optimizer')
                optimizer = tf.train.AdamOptimizer(learning_rate=inputs['learning_rate'])
            else:
                logging.info('use sgd optimizer')
                optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=config['weight_decay'],
                                                              learning_rate=inputs['learning_rate'],
                                                              momentum=config['momentum'])
            if config.get('aggregation_method', 'default') == 'experimental':
                logging.info('use gradient aggregation method: experimental')
                gradient_var_pairs = optimizer.compute_gradients(outputs['total_loss'],
                                                                 var_list=tf.trainable_variables(),
                                                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            else:
                logging.info('use gradient aggregation method: default')
                gradient_var_pairs = optimizer.compute_gradients(outputs['total_loss'],
                                                                 var_list=tf.trainable_variables())
            vars = [x[1] for x in gradient_var_pairs if x[0] is not None]
            gradients = [x[0] for x in gradient_var_pairs if x[0] is not None]
            gc = config.get('gradient_clipping', 120.0)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=gc)
            outputs['train_op'] = optimizer.apply_gradients(zip(gradients, vars))

    return inputs, outputs


def train(args, config, word2idx, model_inputs, model_outputs, sess):
    batch_generator, num_batches = get_batch_generator(args.input_file,
                                                       word2idx,
                                                       config['batch_size'],
                                                       config['max_length'],
                                                       reverse=config['reverse'])
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

    step = 0
    val_ppls = list()
    best_val_ppl = None
    learning_rate = config['learning_rate']
    start = time.time()

    for epoch in range(config['epochs']):
        overall_loss = 0
        total_tokens = 0
        logging.info('epoch: {}'.format(epoch+1))
        for batch_id, (x, y, seq_lengths) in enumerate(batch_generator(do_shuffle=True)):
            step += 1
            feed_dict = {model_inputs['x']: x,
                         model_inputs['y']: y,
                         model_inputs['seq_length']: seq_lengths,
                         model_inputs['input_keep_prob']: config['input_keep_prob'],
                         model_inputs['output_keep_prob']: config['output_keep_prob'],
                         model_inputs['learning_rate']: learning_rate}
            outputs = sess.run(model_outputs, feed_dict)
            overall_loss += outputs['total_loss']
            total_tokens += outputs['num_tokens']

            if batch_id % 50 == 0:
                seconds = int(time.time() - start)
                ppl_total = np.exp(overall_loss / total_tokens)
                ppl_batch = np.exp(outputs['total_loss'] / outputs['num_tokens'])
                logging.info(f'epoch {epoch+1} / {config["epochs"]}, batch {batch_id+1} / {num_batches}, ppl: {ppl_total:.2f} ({ppl_batch:.2f}) (seconds {seconds})')
                start = time.time()

        val_ppl = valid(args, config, word2idx, model_inputs, model_outputs, sess)
        if len(val_ppls) > 1 and val_ppl >= min(val_ppls[-2:]):
            learning_rate = learning_rate * 0.25
            logging.info('change learning rate to {}'.format(learning_rate))
            logging.info(val_ppls)
            val_ppls = list()
            continue
        if best_val_ppl is None or val_ppl < best_val_ppl:
            checkpoint_path = saver.save(sess,
                                         os.path.join(args.save_dir, 'ckpt', 'model.ckpt'),
                                         write_meta_graph=False,
                                         global_step=step)
            logging.info("model saved to {}".format(checkpoint_path))
            best_val_ppl = val_ppl

        val_ppls.append(val_ppl)
        logging.info(val_ppls)

    sess.close()
    logging.info('training finished after {} steps and {} epochs'.format(step, epoch))
    logging.info('best validation loss: {}'.format(best_val_ppl))


def valid(args, config, word2idx, model_inputs, model_outputs, sess):
    overall_loss = 0
    total_tokens = 0
    logging.info('run validation')

    valid_files = [args.valid_file]
    if os.path.isdir(args.valid_file):
        valid_files = [os.path.join(args.valid_file, f) for f in os.listdir(args.valid_file)]

    for valid_file in valid_files:
        batch_generator, num_batches = get_batch_generator(valid_file,
                                                           word2idx,
                                                           config['batch_size'],
                                                           max_length=None,
                                                           reverse=config['reverse'])
        for batch_id, (x, y, seq_lengths) in enumerate(batch_generator(do_shuffle=False)):
            feed_dict = {model_inputs['x']: x,
                         model_inputs['y']: y,
                         model_inputs['seq_length']: seq_lengths}
            total_loss, num_tokens = sess.run([model_outputs['total_loss'], model_outputs['num_tokens']], feed_dict)
            overall_loss += total_loss
            total_tokens += num_tokens

            logging.info(f'batch {batch_id+1} / {num_batches}')
    valid_ppl = np.exp(overall_loss / total_tokens)
    logging.info('validation perplexity: {}'.format(valid_ppl))
    return valid_ppl


def get_batch_generator(input_file, word2idx, batch_size, max_length, reverse=False):
    data = list()
    for line in open(input_file):
        words = [w for w in line.split() if w not in bad_words]
        if max_length is None or len(words) <= max_length:
            data.append([word2idx.get(w, unk_idx) for w in words])
    logging.info('number of instances shorter than {}: {}'.format(max_length, len(data)))
    num_batches = int(len(data) / batch_size)
    logging.info('around {} batches per epoch'.format(num_batches))

    def batch_generator(do_shuffle=True):
        if do_shuffle:
            shuffle(data)
        for batch, offset in enumerate(range(0, len(data), batch_size)):
            #logging.info('batch {}/{}'.format(batch+1, num_batches))
            batch_data = data[offset:offset + batch_size]
            max_len = max(len(d) for d in batch_data)
            real_batch_size = len(batch_data)

            x = np.zeros(shape=(real_batch_size, max_len + 1), dtype=np.int32)
            y = np.zeros(shape=(real_batch_size, max_len + 1), dtype=np.int32)

            for i, sequence in enumerate(batch_data):
                if reverse:
                    sequence = list(reversed(sequence))
                    x[i, :len(sequence) + 1] = [eos_idx] + sequence
                    y[i, :len(sequence) + 1] = sequence + [bos_idx]
                else:
                    x[i, :len(sequence) + 1] = [bos_idx] + sequence
                    y[i, :len(sequence) + 1] = sequence + [eos_idx]

            assert pad_idx == 0
            seq_lengths = np.count_nonzero(x, axis=1)
            yield x, y, seq_lengths
    return batch_generator, num_batches


if __name__ == '__main__':
    main()
