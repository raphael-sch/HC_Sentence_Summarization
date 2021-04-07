Code for ACL 2020 paper: [Schumann et al. "Discrete Optimization for Unsupervised Sentence Summarization with Word-Level Extraction"](https://www.aclweb.org/anthology/2020.acl-main.452.pdf)


Prepare
=======

### Python Version and Requirements
tested with python version 3.7
```
pip install -r requirements.txt
```

### Download Data
Download Gigaword for headline generation from https://github.com/harvardnlp/sent-summary.


Use Model
=========

### Pretrained Weights
Pretrained weights for the language model and the sent2vec embeddings should get cloned automatically with this git repository, otherwise read about Git Large File Storage.

### Run Inference
Set correct paths in `configs/hc_title_8.yaml`
```
python run.py --input_file data/summary/sumdata/Giga/input.txt --config configs/hc_title_8.yaml --output_dir outputs/hc_title_8
```

You can split the input file along its lines to run them in parallel with the arguments `--from_line 100 --to_line 200`

### Evaluate
```
python outputs/evaluate.py --output_dir outputs/hc_title_8/ --reference_file data/summary/sumdata/Giga/task1_ref0.txt
```

#### Results
| Model                 | Rouge-F1-1 | Rouge-F1-2  | Rouge-F1-L |
| --------------------- |:----------:|:-----------:|:----------:|
| HC_title_8 paper      |    26.32   |    9.63     |    24.19   |
| HC_title_8 repository |    26.27   |    9.80     |    24.20   |
| HC_title_10 paper     |    27.52   |   10.27     |    24.91   |
| HC_title_10 repository|    27.07   |   10.12     |    24.65   |



Train Model from Scratch
========================

### Vocabulary file and idf Counts
For custom data you have to create a new vocabulary file with `lm/create_vocab.py` and new idf counts with `cos/tfidf.py`  

### Language Model

```
python lm/language_model.py --input_file path_to_data/train.title.txt --valid_file path_to_data/valid.title.filter.small.txt --vocab_file lm/vocabs/title.vocab --config_file lm/configs/title_forward.yaml --save_dir lm/outputs/title_forward
```
same for backward language model with config `lm/configs/title_backward.yaml`

### Sent2vec
install sent2vec from https://github.com/epfml/sent2vec

```
./fasttext sent2vec -input data/train.title.txt -output title -minCount 5 -dim 700 -epoch 20 -lr 0.2 -wordNgrams 1 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 70000 -numCheckPoints 1
``` 

convert to numpy array with correct vocab order:
```
python scripts/sent2vec_model_to_numpy.py --s2v_model title.bin --vocab_file title.vocab --output_file embeddings/s2v_title.npy
```

Citation
=========
Please cite the following paper if you use this code:

```
@inproceedings{schumann-etal-2020-discrete,
    title = "Discrete Optimization for Unsupervised Sentence Summarization with Word-Level Extraction",
    author = "Schumann, Raphael and Mou, Lili and Lu, Yao and Vechtomova, Olga and Markert, Katja",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    pages = "5032--5042",}
```