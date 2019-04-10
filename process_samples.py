from __future__ import print_function

import time

from nltk.translate.bleu_score import sentence_bleu
import re
import argparse
import torch

from rewards.loader import load_reward_calculator
from utils import read_corpus_de_en
import numpy as np
from scipy.misc import comb
from vocab import Vocab
import math

import gensim

import pickle


# def is_valid_sample(sent):
#     tokens = sent.split(' ')
#     return len(tokens) >= 1 and len(tokens) < 50
#
#
# def sample_from_model(args):
#     para_data = args.parallel_data
#     sample_file = args.sample_file
#     output = args.output
#
#     tgt_sent_pattern = re.compile('^\[(\d+)\] (.*?)$')
#     para_data = [l.strip().split(' ||| ') for l in open(para_data)]
#
#     f_out = open(output, 'w')
#     f = open(sample_file)
#     f.readline()
#     for src_sent, tgt_sent in para_data:
#         line = f.readline().strip()
#         assert line.startswith('****')
#         line = f.readline().strip()
#         print(line)
#         assert line.startswith('target:')
#
#         tgt_sent2 = line[len('target:'):]
#         assert tgt_sent == tgt_sent2
#
#         line = f.readline().strip()  # samples
#
#         tgt_sent = ' '.join(tgt_sent.split(' ')[1:-1])
#         tgt_samples = set()
#         for i in range(1, 101):
#             line = f.readline().rstrip('\n')
#             m = tgt_sent_pattern.match(line)
#
#             assert m, line
#             assert int(m.group(1)) == i
#
#             sampled_tgt_sent = m.group(2).strip()
#
#             if is_valid_sample(sampled_tgt_sent):
#                 tgt_samples.add(sampled_tgt_sent)
#
#         line = f.readline().strip()
#         assert line.startswith('****')
#
#         tgt_samples.add(tgt_sent)
#         tgt_samples = list(tgt_samples)
#
#         assert len(tgt_samples) > 0
#
#         tgt_ref_tokens = tgt_sent.split(' ')
#         bleu_scores = []
#         for tgt_sample in tgt_samples:
#             bleu_score = sentence_bleu([tgt_ref_tokens], tgt_sample.split(' '))
#             bleu_scores.append(bleu_score)
#
#         tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: bleu_scores[i], reverse=True)
#
#         print('%d samples' % len(tgt_samples))
#
#         print('*' * 50, file=f_out)
#         print('source: ' + src_sent, file=f_out)
#         print('%d samples' % len(tgt_samples), file=f_out)
#         for i in tgt_ranks:
#             print('%s ||| %f' % (tgt_samples[i], bleu_scores[i]), file=f_out)
#         print('*' * 50, file=f_out)
#
#     f_out.close()


def get_new_ngram(ngram, n, vocab):
    """
    ngram `ngram`を新しくサンプリングされた同じ長さのngramに置き換える
    replace ngram `ngram` with a newly sampled ngram of the same length
    """

    new_ngram_wids = [np.random.randint(3, len(vocab)) for i in range(n)]
    new_ngram = [vocab.id2word[wid] for wid in new_ngram_wids]

    return new_ngram


def get_reward_calculator(args):
    # TODO: docopt移行
    return load_reward_calculator({
        '--valid-metric': args.reward,
        '--preprocessed-data': args.preprocessed_data,
        '--reward-target-vector': args.target_vec,
        '--smooth-bleu': args.smooth_bleu,
    })


def sample_ngram_sentence(args, word_list, vocab):
    tgt_len = len(word_list)
    n = np.random.randint(1, min(tgt_len,
                                 args.max_ngram_size + 1))  # we do not replace the last token: it must be a period!

    idx = np.random.randint(tgt_len - n)
    ngram = word_list[idx: idx + n]
    new_ngram = get_new_ngram(ngram, n, vocab)

    sampled_tgt_sent = list(word_list)
    sampled_tgt_sent[idx: idx + n] = new_ngram
    return sampled_tgt_sent


def sample_ngram(args):
    src_sents = read_corpus_de_en(args.src, 'src')
    tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>
    f_out = open(args.output, 'w')

    vocab = Vocab.load(args.vocab)
    tgt_vocab = vocab.tgt

    begin_time = time.time()

    reward_calc = get_reward_calculator(args)

    print('sample_ngram', len(src_sents), len(tgt_sents), f_out)
    for i, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        src_sent = ' '.join(src_sent)

        tgt_samples = []

        # generate 100 samples

        # append itself
        tgt_samples.append(tgt_sent)

        for sid in range(args.sample_size - 1):
            sampled_tgt_sent = sample_ngram_sentence(args, tgt_sent, tgt_vocab)

            # compute the probability of this sample
            # prob = 1. / args.max_ngram_size * 1. / (tgt_len - 1 + n) * 1 / (len(tgt_vocab) ** n)

            tgt_samples.append(sampled_tgt_sent)

        # compute bleu scores or edit distances and rank the samples by bleu scores
        rewards = []
        _tgt_sent_vec = None
        for tgt_sample in zip(tgt_samples):
            reward = reward_calc.compute_sentence_reward(tgt_sent, tgt_sample)
            rewards.append(reward)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: rewards[i], reverse=True)
        # convert list of tokens into a string
        tgt_samples = [' '.join(tgt_sample) for tgt_sample in tgt_samples]

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for _i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[_i], rewards[_i]), file=f_out)
        print('*' * 50, file=f_out)

        if i % 1000 == 0 and i != 0:
            print('done %d [%d s].' % (i, time.time() - begin_time))
            if args.is_debug:
                print('debug mode!! stop')
                break;

    f_out.close()


# MARK: word2vec

# def get_new_ngram_word2vec(ngram, vocab, model):
#     """
#     ngram `ngram`を新しくサンプリングされた同じ長さのngramに置き換える
#     replace ngram `ngram` with a newly sampled ngram of the same length
#     """
#
#     def sample_w2v_simlar(word):
#         if word in model.vocab:
#             similar_arr = np.array(model.most_similar(word))
#             words = similar_arr[:, 0]
#             probs = similar_arr[:, 1].astype(np.float64)
#             probs = probs / sum(probs)
#             return np.random.choice(words, p=probs)
#         else:
#             wid = np.random.randint(3, len(vocab))
#             return vocab.id2word[wid]
#
#     new_ngram = [sample_w2v_simlar(w) for w in ngram]
#
#     return new_ngram

def get_w2v_contain_index_list_similar(word_list, model, model_vocab_set):
    valid_words = set(word_list) & model_vocab_set
    similar_dict = {word: np.array(model.most_similar(word)) for word in valid_words}
    valid_words_index = [i for i, w in enumerate(word_list) if w in valid_words]
    return valid_words_index, similar_dict


def sample_w2v_simlar(similar_arr):
    words = similar_arr[:, 0]
    probs = similar_arr[:, 1].astype(np.float64)
    probs = probs / sum(probs)
    return np.random.choice(words, p=probs)


def sample_ngram_word2vec_sentence(args, word_list, valid_words_index, similar_dict):
    """
    一文に対する入れ替え
    変更箇所をmodel.vacabに含まれる単語に制限してみる。

    """

    tgt_len = len(valid_words_index)
    if tgt_len == 0:
        return None

    # ngram
    n = np.random.randint(1, min(tgt_len,
                                 args.max_ngram_size + 1))  # we do not replace the last token: it must be a period!
    idxs = np.random.choice(valid_words_index, size=n, replace=False)

    # 変更ngram実体
    ngram = word_list[idxs]
    # 単語取得
    new_ngram = [sample_w2v_simlar(similar_dict[w]) for w in ngram]

    sampled_tgt_sent = word_list.copy()
    # 入れ替え
    sampled_tgt_sent[idxs] = new_ngram
    return sampled_tgt_sent


def get_w2v_contain_index_list_by_dict(word_list, vocab_set):
    valid_words = set(word_list) & vocab_set
    valid_words_index = [i for i, w in enumerate(word_list) if w in valid_words]
    return valid_words_index


def sample_ngram_word2vec_by_dict(args):
    """
    単語入れ替えにおいて近い意味で入れ替えられるようにword2vecを用いる。
    :param args:
    :return:
    """
    src_sents = read_corpus_de_en(args.src, 'src')
    tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>

    vocab = Vocab.load(args.vocab)
    tgt_vocab = vocab.tgt
    reward_calc = get_reward_calculator(args)

    begin_time = time.time()
    print('dict loading...')
    with open('./data/word2vec/de_en_similar_dict.pickle', 'rb') as f:
        similar_dict = pickle.load(f)
    vocab_set = set(similar_dict.keys())

    print('loaded model [%d s].' % (time.time() - begin_time))

    f_out = open(args.output, 'w')

    print('sample ngram word2vec', len(src_sents), len(tgt_sents), f_out)
    for i, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        tgt_sent = np.array(tgt_sent)
        tgt_sample_list = [tgt_sent]

        # 高速化のため、ここでword2vecに関する情報を整理しておく
        _st = time.time()
        valid_words_index = get_w2v_contain_index_list_by_dict(tgt_sent, vocab_set)
        print(f'get_w2v_contain_index_list_by_dict : [{time.time()-_st:.3g} s].')
        if len(valid_words_index) <= 1:
            print(f'入れ替え可能な単語がありません。:{" ".join(tgt_sent)}')
            continue

        _st = time.time()
        tgt_sample_list += [sample_ngram_word2vec_sentence(args, tgt_sent, valid_words_index, similar_dict) for _ in
                            range(args.sample_size - 1)]
        print(f'sample_ngram_word2vec_sentence : {args.sample_size - 1}samples: [{time.time()-_st:.3g} s].')

        _st = time.time()
        reward_list = []
        for tgt_sample in tgt_sample_list:
            reward = reward_calc.compute_sentence_reward(tgt_sent, tgt_sample)
            reward_list.append(reward)
        print(f'compute rewards : {len(tgt_sample_list)}samples: [{time.time()-_st:.3g} s].')

        tgt_ranks = sorted(range(len(tgt_sample_list)), key=lambda i: reward_list[i], reverse=True)
        tgt_sample_list = [' '.join(tgt_sample) for tgt_sample in tgt_sample_list]

        print('*' * 50, file=f_out)
        src_sent = ' '.join(src_sent)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_sample_list), file=f_out)
        for _i in tgt_ranks:
            print('%s ||| %f' % (tgt_sample_list[_i], reward_list[_i]), file=f_out)
        print('*' * 50, file=f_out)

        if i % 100 == 0 and i != 0:
            print('done %d [%d s].' % (i, time.time() - begin_time))
            if args.is_debug:
                print('debug mode!! stop')
                break;

    f_out.close()


def sample_ngram_word2vec(args):
    """
    単語入れ替えにおいて近い意味で入れ替えられるようにword2vecを用いる。
    :param args:
    :return:
    """
    src_sents = read_corpus_de_en(args.src, 'src')
    tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>

    vocab = Vocab.load(args.vocab)
    tgt_vocab = vocab.tgt
    reward_calc = get_reward_calculator(args)

    begin_time = time.time()
    print('model loading...')
    model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec/GoogleNews-vectors-negative300.bin.gz',
        # './data/word2vec/text8.bin.gz',
        binary=True)
    model_vocab_set = set(model.vocab)

    print('loaded model [%d s].' % (time.time() - begin_time))

    f_out = open(args.output, 'w')

    print('sample ngram word2vec', len(src_sents), len(tgt_sents), f_out)
    for i, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        tgt_sent = np.array(tgt_sent)
        tgt_sample_list = [tgt_sent]

        # 高速化のため、ここでword2vecに関する情報を整理しておく
        _st = time.time()
        valid_words_index, similar_dict = get_w2v_contain_index_list_similar(tgt_sent, model, model_vocab_set)
        print(f'get_w2v_contain_index_list_similar : [{time.time()-_st:.3g} s].')
        if len(valid_words_index) <= 1:
            print(f'入れ替え可能な単語がありません。:{" ".join(tgt_sent)}')
            continue

        _st = time.time()
        tgt_sample_list += [sample_ngram_word2vec_sentence(args, tgt_sent, valid_words_index, similar_dict) for _ in
                            range(args.sample_size - 1)]
        print(f'sample_ngram_word2vec_sentence : {args.sample_size - 1}samples: [{time.time()-_st:.3g} s].')

        _st = time.time()
        reward_list = []
        for tgt_sample in tgt_sample_list:
            reward = reward_calc.compute_sentence_reward(tgt_sent, tgt_sample)
            reward_list.append(reward)
        print(f'compute rewards : {len(tgt_sample_list)}samples: [{time.time()-_st:.3g} s].')

        tgt_ranks = sorted(range(len(tgt_sample_list)), key=lambda i: reward_list[i], reverse=True)
        tgt_sample_list = [' '.join(tgt_sample) for tgt_sample in tgt_sample_list]

        print('*' * 50, file=f_out)
        src_sent = ' '.join(src_sent)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_sample_list), file=f_out)
        for _i in tgt_ranks:
            print('%s ||| %f' % (tgt_sample_list[_i], reward_list[_i]), file=f_out)
        print('*' * 50, file=f_out)

        if i % 100 == 0 and i != 0:
            print('done %d [%d s].' % (i, time.time() - begin_time))
            if args.is_debug:
                print('debug mode!! stop')
                break;

    f_out.close()


# MARK : hamming_distance
def sample_hamming_distance(args):
    src_sents = read_corpus_de_en(args.src, 'src')
    tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>

    vocab = Vocab.load(args.vocab)
    tgt_vocab = vocab.tgt
    reward_calc = get_reward_calculator(args)

    f_out = open(args.output, 'w')

    begin_time = time.time()

    print('sample_hamming_distance', len(src_sents), len(tgt_sents), f_out)
    for i, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        src_sent = ' '.join(src_sent)
        tgt_sample_list = [tgt_sent]
        tgt_sample_list += [sample_sentence_hamming_distance(args, tgt_sent, tgt_vocab) for _ in
                            range(args.sample_size - 1)]
        reward_list = []
        for tgt_sample in tgt_sample_list:
            reward = reward_calc.compute_sentence_reward(tgt_sent, tgt_sample)
            reward_list.append(reward)

        tgt_ranks = sorted(range(len(tgt_sample_list)), key=lambda i: reward_list[i], reverse=True)
        tgt_sample_list = [' '.join(tgt_sample) for tgt_sample in tgt_sample_list]

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_sample_list), file=f_out)
        for _i in tgt_ranks:
            print('%s ||| %f' % (tgt_sample_list[_i], reward_list[_i]), file=f_out)
        print('*' * 50, file=f_out)

        if i % 1000 == 0 and i != 0:
            print('done %d [%d s].' % (i, time.time() - begin_time))
            if args.is_debug:
                print('debug mode!! stop')
                break;

    f_out.close()


def sample_sentence_hamming_distance(args, word_list, vocab):
    # https://github.com/isl-mt/xnmt-isl/blob/c526225188d94ea3595476e05354de8d33ebd31a/xnmt/input_readers.py#L229-L277
    length = len(word_list)
    logits = np.arange(length) * (-1) * args.temp  # -n/tau ?
    logits = np.exp(logits - np.max(logits))  # e^(-n/tau - max(-n/tau))
    probs = logits / np.sum(logits)  # p(n) = e^(-n/tau)/Z
    num_words = np.random.choice(length, p=probs)  # 入れ替える単語数 確率用
    corrupt_pos = np.random.binomial(1, p=num_words / length, size=(length,))  # その位置を変えるか否か[1, 0, 1..]
    num_words_to_sample = np.sum(corrupt_pos)  # 入れ替える単語数 決定
    sampled_words = np.random.choice(np.arange(2, len(vocab)), size=(num_words_to_sample,))  # 有効な文字indexをランダムにサンプリング

    new_words = [vocab.id2word[wid] for wid in np.random.randint(3, len(vocab), size=len(sampled_words))]
    corrupt_pos_index = np.where(corrupt_pos == 1)[0].tolist()

    res_word_list = np.array(word_list)
    res_word_list[corrupt_pos_index] = new_words

    return res_word_list


# def sample_ngram_adapt(args):
#     src_sents = read_corpus_de_en(args.src, 'src')
#     tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>
#     f_out = open(args.output, 'w')
#
#     vocab = torch.load(args.vocab)
#     tgt_vocab = vocab.tgt
#
#     max_len = max([len(tgt_sent) for tgt_sent in tgt_sents]) + 1
#
#     for src_sent, tgt_sent in zip(src_sents, tgt_sents):
#         src_sent = ' '.join(src_sent)
#
#         tgt_len = len(tgt_sent)
#         tgt_samples = []
#
#         # generate 100 samples
#
#         # append itself
#         tgt_samples.append(tgt_sent)
#
#         for sid in range(args.sample_size - 1):
#             max_n = min(tgt_len - 1, 4)
#             bias_n = int(max_n * tgt_len / max_len) + 1
#             assert 1 <= bias_n <= 4, 'bias_n={}, not in [1,4], max_n={}, tgt_len={}, max_len={}'.format(bias_n, max_n,
#                                                                                                         tgt_len,
#                                                                                                         max_len)
#
#             p = [1.0 / (max_n + 5)] * max_n
#             p[bias_n - 1] = 1 - p[0] * (max_n - 1)
#             assert abs(sum(p) - 1) < 1e-10, 'sum(p) != 1'
#
#             n = np.random.choice(np.arange(1, int(max_n + 1)),
#                                  p=p)  # we do not replace the last token: it must be a period!
#             assert n < tgt_len, 'n={}, tgt_len={}'.format(n, tgt_len)
#
#             idx = np.random.randint(tgt_len - n)
#             ngram = tgt_sent[idx: idx + n]
#             new_ngram = get_new_ngram(ngram, n, tgt_vocab)
#
#             sampled_tgt_sent = list(tgt_sent)
#             sampled_tgt_sent[idx: idx + n] = new_ngram
#
#             tgt_samples.append(sampled_tgt_sent)
#
#         # compute bleu scores and rank the samples by bleu scores
#         bleu_scores = []
#         for tgt_sample in tgt_samples:
#             bleu_score = sentence_bleu([tgt_sent], tgt_sample)
#             bleu_scores.append(bleu_score)
#
#         tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: bleu_scores[i], reverse=True)
#         # convert list of tokens into a string
#         tgt_samples = [' '.join(tgt_sample) for tgt_sample in tgt_samples]
#
#         print('*' * 50, file=f_out)
#         print('source: ' + src_sent, file=f_out)
#         print('%d samples' % len(tgt_samples), file=f_out)
#         for i in tgt_ranks:
#             print('%s ||| %f' % (tgt_samples[i], bleu_scores[i]), file=f_out)
#         print('*' * 50, file=f_out)
#
#     f_out.close()


# def sample_from_hamming_distance_payoff_distribution(args):
#     src_sents = read_corpus_de_en(args.src, 'src')
#     tgt_sents = read_corpus_de_en(args.tgt, 'src')  # do not read in <s> and </s>
#     f_out = open(args.output, 'w')
#
#     vocab = torch.load(args.vocab)
#     tgt_vocab = vocab.tgt
#
#     payoff_prob, Z_qs = generate_hamming_distance_payoff_distribution(max(len(sent) for sent in tgt_sents),
#                                                                       vocab_size=len(vocab.tgt),
#                                                                       tau=args.temp)
#
#     for src_sent, tgt_sent in zip(src_sents, tgt_sents):
#         tgt_samples = []  # make sure the ground truth y* is in the samples
#         tgt_sent_len = len(tgt_sent) - 3  # remove <s> and </s> and ending period .
#         tgt_ref_tokens = tgt_sent[1:-1]
#         bleu_scores = []
#
#         # sample an edit distances
#         e_samples = np.random.choice(range(tgt_sent_len + 1), p=payoff_prob[tgt_sent_len], size=args.sample_size,
#                                      replace=True)
#
#         for i, e in enumerate(e_samples):
#             if e > 0:
#                 # sample a new tgt_sent $y$
#                 old_word_pos = np.random.choice(range(1, tgt_sent_len + 1), size=e, replace=False)
#                 new_words = [vocab.tgt.id2word[wid] for wid in np.random.randint(3, len(vocab.tgt), size=e)]
#                 new_tgt_sent = list(tgt_sent)
#                 for pos, word in zip(old_word_pos, new_words):
#                     new_tgt_sent[pos] = word
#
#                 bleu_score = sentence_bleu([tgt_ref_tokens], new_tgt_sent[1:-1])
#                 bleu_scores.append(bleu_score)
#             else:
#                 new_tgt_sent = list(tgt_sent)
#                 bleu_scores.append(1.)
#
#             # print('y: %s' % ' '.join(new_tgt_sent))
#             tgt_samples.append(new_tgt_sent)
#
#
# def generate_hamming_distance_payoff_distribution(max_sent_len, vocab_size, tau=1.):
#     """compute the q distribution for Hamming Distance (substitution only) as in the RAML paper"""
#     probs = dict()
#     Z_qs = dict()
#     for sent_len in range(1, max_sent_len + 1):
#         counts = [1.]  # e = 0, count = 1
#         for e in range(1, sent_len + 1):
#             # apply the rescaling trick as in https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
#             count = comb(sent_len, e) * math.exp(-e / tau) * ((vocab_size - 1) ** (e - e / tau))
#             counts.append(count)
#
#         Z_qs[sent_len] = Z_q = sum(counts)
#         prob = [count / Z_q for count in counts]
#         probs[sent_len] = prob
#
#         # print('sent_len=%d, %s' % (sent_len, prob))
#
#     return probs, Z_qs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sample_ngram', 'sample_hamming_distance', 'sample_ngram_word2vec'], required=True)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    parser.add_argument('--preprocessed_data', type=str)
    parser.add_argument('--target_vec', type=str)
    parser.add_argument('--parallel_data', type=str)
    parser.add_argument('--sample_file', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--reward', choices=['bleu', 'edit_dist', 'lda', 'deviation', 'deviation_diff', 'shorten'],
                        default='bleu')
    parser.add_argument('--max_ngram_size', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--smooth_bleu', action='store_true', default=False)
    parser.add_argument('--is_debug', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'sample_ngram':
        sample_ngram(args)
    elif args.mode == 'sample_hamming_distance':
        sample_hamming_distance(args)
    elif args.mode == 'sample_ngram_word2vec':
        sample_ngram_word2vec_by_dict(args)

    # elif args.mode == 'sample_from_model':
    #     sample_from_model(args)
    # elif args.mode == 'sample_ngram_adapt':
    #     sample_ngram_adapt(args)
