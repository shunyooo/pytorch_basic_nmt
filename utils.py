import json
import math
import re
import sys
from typing import List, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import slack
from rewards.utils import preprocessing, euclid_sim

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus_de_en(file_path, source):
    """
    de-en用のもの
    :param file_path:
    :param source:
    :return:
    """
    data = []
    for line in open(file_path):
        sent = line.strip()
        sent = preprocessing(sent)
        sent = sent.split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = [SENTENCE_START] + sent + [SENTENCE_END]
        data.append(sent)

    return data

def abstract2sents(abstract):
    """Splits abstract text from datafile into list of sentences.
    Args:
      abstract: string containing <s> and </s> tags for starts and ends of sentences
    Returns:
      sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def read_corpus_cnndaily(file_path):
    """
    cnndailyコーパスを読み込む。
    abstractは一文として読み込む。
    :param file_path:
    :return: src_data, tgt_data
    """
    print(f'read_corpus:{file_path}', file=sys.stderr)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    src_data = [s['article'].split(' ') for s in data]
    tgt_data = [[SENTENCE_START]+''.join(abstract2sents(s['abstract'])).split(' ')+[SENTENCE_END] for s in data]
    return src_data, tgt_data


def read_raml_train_data(data_file, temp):
    train_data = dict()
    num_pattern = re.compile('^(\d+) samples$')
    with open(data_file) as f:
        while True:
            line = f.readline()
            if line is None or line == '':
                break

            assert line.startswith('***')

            src_sent = f.readline()[len('source: '):].strip()
            tgt_num = int(num_pattern.match(f.readline().strip()).group(1))
            tgt_samples = []
            tgt_scores = []
            for i in range(tgt_num):
                d = f.readline().strip().split(' ||| ')
                if len(d) < 2:
                    continue

                tgt_sent = d[0].strip()
                bleu_score = float(d[1])
                tgt_samples.append(tgt_sent)
                tgt_scores.append(bleu_score / temp)

            tgt_scores = np.exp(tgt_scores)
            tgt_scores = tgt_scores / np.sum(tgt_scores)

            tgt_entry = list(zip(tgt_samples, tgt_scores))
            train_data[src_sent] = tgt_entry

            line = f.readline()

    return train_data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss


# MARK: train logや通知用
def list_dict_update(data_dict, add_dict, mode, is_save=False):
    """
    data_dictにadd_dictを結合する。
    data_dictのvalueはlist, add_dictのvalueはスカラ, strの前提
    mode = train, valid, test
    """

    _small_data_dict = None
    if mode in data_dict:
        _small_data_dict = data_dict[mode]
    else:
        data_dict[mode] = {}
        _small_data_dict = data_dict[mode]

    for k, v in add_dict.items():
        if k in _small_data_dict:
            _small_data_dict[k].append(v)
        else:
            _small_data_dict[k] = [v]

    if 'args' in data_dict:
        if is_save:
            file_path = data_dict['args']['--log-data']
            print(f'log_data save to {file_path}')
            with open(file_path, 'wb') as log_out:
                pickle.dump(data_dict, log_out)
    else:
        raise Exception('ERROR: argsをlog_dataに入れておいてください')


def notify_slack_if_need(text, args):
    if args['--notify-slack']:
        slack.post(text)


def log_decode_to_tensorboard(global_step, log_indexes, writer, dev_data=None, eval_info=None):
    for _i in log_indexes:
        text = ''
        if global_step < 0:
            _input = ' '.join(dev_data[_i][0])
            text = f'''| Input |\n| ---------- |\n| {_input} |\n'''
            print(f'log_decode_to_tensorboard input: {text}')
        else:
            hypo = eval_info['top_hyps'][_i]
            text = f'''| Text    | Hypo Score |\n| ------- | ---- |\n| {hypo2str(hypo)} | {hypo.score} |\n'''
        writer.add_text(f'top_hypos【{_i}】', text, global_step)


def log_decode_to_tensorboard_raml(global_step, log_indexes, writer, reward_calc, args=None, dev_data=None,
                                   eval_info=None):
    np.set_printoptions(precision=3, floatmode='maxprec')
    metric = args['--valid-metric']
    for _i in log_indexes:
        text = ''
        _input, _tgt = dev_data[_i]
        _input_str = ' '.join(_input)
        if global_step < 0:
            score = reward_calc.compute_sentence_reward(_tgt, _input)
            text = f'''| Input | {metric} |\n| ---------- | ---------- |\n| {_input_str} | {score} |'''
            print(f'log_decode_to_tensorboard input: {text}')
        else:
            hypo = eval_info['top_hyps'][_i]
            score = reward_calc.compute_sentence_reward(_tgt, hypo.value)
            text = f'''| Text | {metric} | Hypo Score |\n| ------- | ---- | ---- |\n| {hypo2str(hypo)} | {score} | {hypo.score} | '''
        writer.add_text(f'top_hypos【{_i}】', text, global_step)


def hypo2str(hypo):
    return f"{' '.join(hypo.value)}"


def remove_s_tag(word_list):
    if word_list[0] == '<s>':
        word_list = word_list[1:-1]
    return word_list
