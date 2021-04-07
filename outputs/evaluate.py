import os
import argparse

from pythonrouge.pythonrouge import Pythonrouge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--reference_file', type=str)
    args = parser.parse_args()

    lines_dir = os.path.join(args.output_dir, 'lines')

    summaries_file = os.path.join(args.output_dir, 'summaries.txt')
    with open(summaries_file, 'w') as f_out:
        line_ids = sorted(map(int, os.listdir(lines_dir)))
        for line_id in line_ids:
            summary_file = os.path.join(lines_dir, str(line_id), 'line.txt')
            with open(summary_file) as f_in:
                summary = f_in.readline()
                f_out.write(summary)

    all_references = [[l.split() for l in open(args.reference_file).read().splitlines()]]
    result_sentences = [l.split() for l in open(summaries_file)]

    r = get_rouge_perl(result_sentences, all_references)
    print('F1')
    print(r['r1_f1_mid'])
    print(r['r2_f1_mid'])
    print(r['rL_f1_mid'])


def get_rouge_perl(summaries, all_references):
    summary = [[' '.join(s)] for s in summaries]
    reference = [[] for _ in range(len(summary))]
    for references in all_references:
        for i, r in enumerate(references):
            reference[i].append([' '.join(r)])
    # assert len(summary) == len(all_references[0])
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        stemming=True, stopwords=False,
                        word_level=False, length_limit=False, length=100,
                        use_cf=True, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    scores = rouge.calc_score()
    r = dict()
    for score_type in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        r_type = score_type[0].lower() + score_type[-1]
        r['{}_f1_mid'.format(r_type)] = scores['{}-F'.format(score_type)]
        # r['{}_f1_low'.format(r_type)] = scores['{}-F-cf95'.format(score_type)][0]
        # r['{}_f1_high'.format(r_type)] = scores['{}-F-cf95'.format(score_type)][1]

        r['{}_recall_mid'.format(r_type)] = scores['{}-R'.format(score_type)]
        # r['{}_recall_low'.format(r_type)] = scores['{}-R-cf95'.format(score_type)][0]
        # r['{}_recall_high'.format(r_type)] = scores['{}-R-cf95'.format(score_type)][1]
    return {k: v * 100 for k, v in r.items()}


if __name__ == '__main__':
    main()
