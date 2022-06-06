from transformers import PegasusTokenizer
from transformers import PegasusForConditionalGeneration
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class CTRLEval:
    def __init__(self, iwf_dir=None, prompt_dir=None, verbal_dir=None,
                 device='cuda', model_name_or_path='google/pegasus-large'):
        # inverse word frequency (IWF) for each token
        with open(iwf_dir, 'r') as f_iwf:
            self.iwf_score = [float(line.strip()) for line in f_iwf.readlines()]

        # load prompts for attribute relevance
        with open(prompt_dir, 'r') as f_pr:
            self.prompt_list = [line.strip() for line in f_pr.readlines()]

        # load verbalizers for attribute relevance
        self.verbal_list = []
        with open(verbal_dir, 'r') as f_veb:
            line_id = 0
            for line in f_veb.readlines():
                if line_id == 0:
                    # first line: label name
                    self.label_name = line.strip().split('\t')
                else:
                    # following lines: label word (verbalizer) for each label
                    self.verbal_list.append(line.strip().split('\t'))
                line_id += 1

        # device: cuda (for gpu) / cpu
        self.device = device
        # model_name_or_path: model name or the path for the downloaded pre-trained model
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name_or_path)
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.model.config.pad_token_id)

    def lm_score(self, src_text, tgt_text, has_iwf=True, add_special_tokens=True):
        # compute the log probability of pre-trained models
        batch = self.tokenizer(src_text, truncation=True, padding='longest',
                               return_tensors="pt").to(self.device)
        labels = self.tokenizer(tgt_text, truncation=True, padding='longest', add_special_tokens=add_special_tokens,
                                return_tensors="pt").to(self.device)

        # use IWF scores as weights for coherence and consistency
        if has_iwf:
            tgt_score = [max([self.iwf_score[token_id] for token_id in
                              labels['input_ids'][label_id].cpu().numpy()]) for label_id in
                         range(labels['input_ids'].shape[0])]
        else:
            tgt_score = []

        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            labels=labels['input_ids'])
        logits = output.logits.view(-1, self.model.config.vocab_size)
        loss = self.loss_fct(logits, labels['input_ids'].view(-1))
        tgt_len = labels['attention_mask'].sum(dim=1)
        loss = loss.view(labels['input_ids'].shape[0], -1)
        loss = loss.sum(dim=1) / tgt_len

        return loss, tgt_score

    def coh_score(self, data, batch_size):
        # coherence
        data_split = [sent_tokenize(data_ele) for data_ele in data]

        def get_mask_data(data_list):
            # mask each sentence respectively
            src_list, tgt_list, len_list = [], [], []
            for data_ele in data_list:
                src_list_ele, tgt_list_ele = [], []
                for idx in range(len(data_ele)):
                    tgt_list_ele.append(data_ele[idx])
                    src_list_ele.append(' '.join(data_ele[:idx]) + ' <mask_1> ' + ' '.join(data_ele[idx + 1:]))
                src_list.extend(src_list_ele)
                tgt_list.extend(tgt_list_ele)
                len_list.append(len(data_ele))
            return src_list, tgt_list, len_list

        # data_len: list of the number of sentences in each generated result
        src_data, tgt_data, data_len = get_mask_data(data_split)

        # eval_score: score of each pattern evaluator
        # beta: (unnormalized) weight factor of each pattern evaluator
        eval_score, beta = [], []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, tgt_score = self.lm_score(src_text, tgt_text)
                cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]

            eval_score.extend(cur_score)
            beta.extend(tgt_score)

        # compute final score via the weighted sum of pattern evaluators
        data_st = 0
        res_score = []
        for len_ele in data_len:
            if sum(beta[data_st: data_st + len_ele]) > 0:
                res_score.append(np.dot(eval_score[data_st: data_st + len_ele], beta[data_st: data_st + len_ele]) /
                                 sum(beta[data_st: data_st + len_ele]))
            else:
                res_score.append(np.mean(eval_score[data_st: data_st + len_ele]))
            data_st += len_ele

        return res_score

    def cons_score(self, data, prefix, batch_size):
        # consistency

        def get_mask_data(data_list, prefix_list):
            # mask the prefix and generated result respectively
            src_list, tgt_list, len_list = [], [], []
            for data_ele, prefix_ele in zip(data_list, prefix_list):
                assert data_ele.index(prefix_ele) == 0
                src_list_ele = [prefix_ele + ' <mask_1>', '<mask_1> ' + data_ele[len(prefix_ele):]]
                tgt_list_ele = [data_ele[len(prefix_ele):], prefix_ele]
                src_list.extend(src_list_ele)
                tgt_list.extend(tgt_list_ele)
                len_list.append(2)
            return src_list, tgt_list, len_list

        src_data, tgt_data, data_len = get_mask_data(data, prefix)

        # eval_score: score of each pattern evaluator
        # beta: (unnormalized) weight factor of each pattern evaluator
        eval_score, beta = [], []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, tgt_score = self.lm_score(src_text, tgt_text, add_special_tokens=False)
                cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]

            eval_score.extend(cur_score)
            beta.extend(tgt_score)

        # compute final score via the weighted sum of pattern evaluators
        data_st = 0
        res_score = []
        for len_ele in data_len:
            if sum(beta[data_st: data_st + len_ele]) > 0:
                res_score.append(np.dot(eval_score[data_st: data_st + len_ele], beta[data_st: data_st + len_ele]) /
                                 sum(beta[data_st: data_st + len_ele]))
            else:
                res_score.append(np.mean(eval_score[data_st: data_st + len_ele]))
            data_st += len_ele

        return res_score

    def ar_score(self, data, label_str, batch_size):
        # attribute relevance
        label = [self.label_name.index(label_ele) for label_ele in label_str]

        def get_mask_data(data_list, prompt_list, verbal_list):
            # use prompts and verbalizers to generate data
            src_list, tgt_list, len_list = [], [], []
            for data_ele in data_list:
                src_list_ele, tgt_list_ele = [], []
                for idx in range(len(prompt_list)):
                    for idy in range(len(verbal_list)):
                        for idz in range(len(verbal_list[0])):
                            src_list_ele.append(prompt_list[idx].replace('<gen_result>',
                                                                         data_ele).replace('<mask_token>', '<mask_1>'))
                            tgt_list_ele.append(verbal_list[idy][idz])
                src_list.extend(src_list_ele)
                tgt_list.extend(tgt_list_ele)
            return src_list, tgt_list

        src_data, tgt_data = get_mask_data(data, self.prompt_list, self.verbal_list)

        # eval_score: LM score for each pair of prompts and verbalizers
        eval_score, beta = [], []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, _ = self.lm_score(src_text, tgt_text, has_iwf=False, add_special_tokens=False)
                cur_score = [torch.exp(-loss_ele).detach().cpu().numpy() for loss_ele in loss]

            eval_score.extend(cur_score)

        score_pair = np.reshape(eval_score, (-1, len(self.verbal_list[0])))
        # compute unnormalized weight scores
        weight_unnormal = np.sum(score_pair, axis=1)
        # compute the score of each pattern evaluator
        score_pair /= np.sum(score_pair, axis=1, keepdims=True)
        score_data = np.reshape(score_pair, (-1, len(self.prompt_list) * len(self.verbal_list), len(self.verbal_list[0])))
        weight_unnormal = np.reshape(weight_unnormal, (-1, len(self.prompt_list) * len(self.verbal_list)))
        # compute normalized weight scores
        weight_normal = weight_unnormal / np.sum(weight_unnormal, axis=1, keepdims=True)
        weight_normal = np.expand_dims(weight_normal, axis=2)
        res_score = np.choose(np.array(label), np.sum(score_data * weight_normal, axis=1).T)

        return res_score

    def score(self, aspect, data, prefix=None, label=None, batch_size=1):
        # aspect: coh (coherence), cons (consistency), or ar (attribute relevance)
        # data: list of generated texts
        # prefix: list of content prefixes
        # label: list of attribute labels
        if aspect == 'coh':
            return self.coh_score(data, batch_size)
        else:
            if aspect == 'cons':
                return self.cons_score(data, prefix, batch_size)
            else:
                return self.ar_score(data, label, batch_size)
