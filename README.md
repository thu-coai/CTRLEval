# CTRLEval: An Unsupervised Reference-Free Metric for Evaluating Controlled Text Generation

## Introduction

CTRLEval is an unsupervised reference-free metric to evaluate the coherence, consistency, and attribute relevance in controlled text generation tasks. You can read our [paper](https://aclanthology.org/2022.acl-long.164/) for more details. This project is a PyTorch implementation of our work.

## Dependencies

* Python 3.7
* NumPy
* PyTorch 1.4.0
* Transformers (Huggingface) 4.4.2
* NLTK 3.5
* SciPy
* tqdm

## Data

We build an evaluation set for controlled text generation, which is stored in `./data`. The data files `data_senti.txt` and `data_topic.txt` which are used in sentiment-controlled and topic-controlled text generation, respectively, have the same format. Specifically, attribute labels, content prefixes, generated texts, coherence scores, consistency scores, and attribute relevance scores are separated by `\t` in each line.

## Prompt and Verbalizer

For attribute relevance, we design additional prompts and verbalizers, which are stored in `./prompt`. In `prompt_senti.txt` and `prompt_topic.txt`, each line contains a prompt including two special placeholders, i.e., `<gen_result>` (for generated texts) and `<mask_token>` (for mask tokens). In `verbal_senti.txt` and `verbal_topic.txt`, the first line (separated by `\t`) contains the name of each attribute label, where each of the following lines indicates a sequence of label words. You can refer to Section 4.2 and Appendix A in our [paper]((https://aclanthology.org/2022.acl-long.164/)) for more details.

## Quick Start for Evaluation with CTRLEval

We have already built `main.py` to reproduce our overall results. You can directly run this code to obtain the evaluation results:

```shell
python main.py
```

You can also refer to this code to evaluate your own generated texts with CTRLEval.  We give an example (note: you should first enter the directory of CTRLEval):

```python
>>> from ctrleval import CTRLEval
>>> task = 'senti' # evaluation for sentiment-controlled text generation
>>> scorer = CTRLEval(iwf_dir='iwf_full.txt', prompt_dir='./prompt/prompt_{}.txt'.format(task), verbal_dir='./prompt/verbal_{}.txt'.format(task), model_name_or_path='google/pegasus-large')
>>> data = ['The book is about NLP. It depicts fancy models.']
>>> prefix = ['The book']
>>> label = ['positive']
>>> scorer.score(aspect='coh', data=data, batch_size=1) # evaluation of coherence
[-4.723989181567967]
>>> scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1) # evaluation of consistency
[-7.385560478830747]
>>> scorer.score(aspect='ar', data=data, label=label, batch_size=1) # evaluation of attribute relevance
array([0.8916258], dtype=float32)
```

## Disclaimer

The evaluation set aims to facilitate the research for evaluating controlled text generation. The texts in this evaluation set are generated from four representative models including [CTRL](https://github.com/salesforce/ctrl), [PPLM](https://github.com/uber-research/PPLM), [GeDi](https://github.com/salesforce/GeDi), and [CoCon](https://github.com/alvinchangw/COCON_ICLR2021). We directly use the hyper-parameters of original papers to generate samples. Although we asked annotators to check whether there are inappropriate contents in the data , there is no guarantee that all the inappropriate samples have been filtered out. All the contents contained in this evaluation set do not represent the authors' opinion.

## Citation

```
@inproceedings{ke-etal-2022-ctrleval,
    title = "{CTRLE}val: An Unsupervised Reference-Free Metric for Evaluating Controlled Text Generation",
    author = "Ke, Pei  and Zhou, Hao  and Lin, Yankai  and Li, Peng  and Zhou, Jie  and Zhu, Xiaoyan  and Huang, Minlie",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    pages = "2306--2319",
}
```

Please kindly cite our paper if this paper and the codes are helpful.
