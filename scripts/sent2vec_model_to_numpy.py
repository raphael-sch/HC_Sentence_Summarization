import argparse
import numpy as np
import sent2vec

pad_token = 'PAD'
unk_token = 'UNK'
bos_token = 'BOS'
eos_token = 'EOS'
pad_idx = 0
unk_idx = 1
bos_idx = 2
eos_idx = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2v_model', type=str, default='retrained_title.bin')
    parser.add_argument('--vocab_file', type=str, default='title.vocab')
    parser.add_argument('--output_file', type=str, default='embeddings/s2v_title.npy')
    args = parser.parse_args()

    model = sent2vec.Sent2vecModel()
    model.load_model(args.s2v_model, inference_mode=True)
    model.release_shared_mem(args.s2v_model)

    word2idx, idx2word = get_vocabs(args.vocab_file)

    embeddings = np.zeros(shape=(len(word2idx), model.get_emb_size()), dtype=np.float32)
    words_not_found = list()
    for word, i in word2idx.items():
        if word == 'UNK':  # Gigaword dataset has <unk> in train and UNK in test, but should not change results
            word = '<unk>'
        e = model.embed_unigrams([word])
        embeddings[i] = e
        if e.all() == 0 and word != 'PAD':
            words_not_found.append(word)

    print('words_not_found', words_not_found)  # does not matter
    np.save(args.output_file, embeddings)


def get_vocabs(vocab_file):
    word2idx = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
    for line in open(vocab_file):
        word = line.strip().split()[0]
        word2idx[word] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    assert word2idx[pad_token] == pad_idx
    assert word2idx[unk_token] == unk_idx
    assert word2idx[bos_token] == bos_idx
    assert word2idx[eos_token] == eos_idx
    return word2idx, idx2word
