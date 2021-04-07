input_file = '../data/summary/sumdata/train/train.title.txt'
vocab_out_file = 'vocabs/title.vocab'

word_count = dict()

with open(input_file) as f:
    for line in f:
        for word in line.rstrip().split():
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

print('Vocab has {} words'.format(len(word_count)))

with open(vocab_out_file, 'w') as f:
    for word in sorted(word_count, key=lambda w: word_count[w], reverse=True):
        f.write(word + '\n')
