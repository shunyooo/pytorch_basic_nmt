import pickle
import re
import html
from gensim import corpora
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import ldamodel

from .cons import DEVIATION


# --- MARK: 読み込み, 前処理系 ---

def preprocessing(sentence):
    """
    前処理。
    """
    # URL削除
    sentence = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", sentence)
    # sentence = normalize_number(sentence)  # 数字を0に変換
    sentence = sentence.lower()  # アルファベットを小文字に
    sentence = html.unescape(sentence)
    sentence = sentence.strip()
    return sentence


def generate_corpus(texts, no_below, no_above):
    """
    Summary: 辞書、コーパスの生成
             辞書は    [{id  単語  df},...] の情報を持つ。
             コーパスは [(id, df), (id, df), ...]という形式で保存される。
    Attributes:
        @param (texts): 文書。[["word","word","word"], ["word","word","word"]]
        @param (no_below) default=2: no_below以下の出現回数の単語は無視。
        @param (no_above) default=0.3: no_above以上の文書割合に出現したワードはありふれているので、無視
    Returns: 辞書、コーパス
    """
    # 辞書
    dictionary = corpora.Dictionary(texts)
    # 辞書、前処理
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # コーパス
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


def read_text(path, split=True):
    f = open(path)
    lines = [preprocessing(line) for line in f]
    if split:
        lines = [line.split() for line in lines]
    return lines


def generate_topic_doc_indexs(model, corpus):
    # トピックごとの文書頻度
    topic_text_indexs = [[] for _ in range(model.num_topics)]
    for index, bow in enumerate(corpus):
        topics = model.get_document_topics(bow)
        top_topic = sorted(topics, key=lambda t: t[1], reverse=True)[0][0]
        topic_text_indexs[top_topic].append(index)
    return topic_text_indexs


def show_topic_df(model, corpus):
    topic_text_indexs = generate_topic_doc_indexs(model, corpus)
    # トピックごとの頻度分布。
    plt.title("トピックごとの文書頻度")
    plt.bar(range(model.num_topics), [len(indexs) for indexs in topic_text_indexs])
    plt.show()
    return topic_text_indexs


def parse_pos_texts(path, save_to):
    """ 品詞解析 """
    texts = read_text(path, split=False)
    pos_tag_list = []
    with StanfordCoreNLP('http://localhost', port=9000) as nlp:
        for text in tqdm(texts):
            pos_tag_list.append(nlp.pos_tag(text))

    with open(save_to, 'wb') as f:
        pickle.dump(pos_tag_list, f)


def gen_df_lda_top_count(topic_doc_indexs, model, dictionary, corpus):
    word_topics = [[0 for _ in range(model.num_topics)] for _ in range(len(dictionary))]
    for topic, doc_indexs in enumerate(topic_doc_indexs):
        for doc_i in doc_indexs:
            for _id, tf in corpus[doc_i]:
                word_topics[_id][topic] += 1

    df = pd.DataFrame(word_topics, index=dictionary.values())
    df["sum"] = df.apply(lambda row: sum(row), axis=1)

    return df


def _cal_lda_topic_prob_outer(model):
    def do(row):
        word = row.name
        probs = model.get_term_topics(word, minimum_probability=0)
        res_probs = np.zeros(model.num_topics)
        for i, prob in probs:
            res_probs[i] = prob
        return pd.Series(res_probs)

    return do


def add_df_lda_topic_prob(df, model):
    _cal_lda_topic_prob = _cal_lda_topic_prob_outer(model)
    df[["LDA{}".format(t) for t in range(model.num_topics)]] = df.apply(_cal_lda_topic_prob, axis=1)
    return df


# --- MARK: まとめ系 ---

import time
from collections import namedtuple

RWData = namedtuple('RWData', ('dictionary', 'corpus', 'model', 'topic_doc_indexs', 'df_word', 'df_topic'))


def _run(f, info=""):
    st_time = time.time()
    print(f'{info}...', end="")
    res = f()
    print(f'done! {int(time.time() - st_time)}sec')
    return res


def generate_df_lda_deviation_from_texts(texts, topic_N, no_below=2, no_above=0.6):
    """
    テキストから色々丸ごと構築する
    :param texts: [['word', 'word',..], ['word', 'word',..]]
    :param topic_N: トピック数
    :param no_below: no_below以下の出現単語を除く
    :param no_above: no_above以上の出現率の単語を除く
    :return:
    """
    _info = f'generate_corpus from {len(texts)} docs'
    dictionary, corpus = _run(lambda: generate_corpus(texts, no_below=no_below, no_above=no_above), _info)

    _info = f'make LDA model, topic:{topic_N}, terms:{len(dictionary)}'
    model = _run(lambda: ldamodel.LdaModel(corpus=corpus, num_topics=topic_N, id2word=dictionary), _info)

    _info = f'split documents to topic[doc, doc, doc, ]'
    topic_doc_indexs = _run(lambda: generate_topic_doc_indexs(model, corpus), _info)

    _info = f'generate DataFrame for word'
    df_word = _run(lambda: gen_df_lda_top_count(topic_doc_indexs, model, dictionary, corpus), _info)

    _info = f'add DataFrame LDA[topic] for word'
    df_word = _run(lambda: add_df_lda_topic_prob(df_word, model), _info)

    _info = f'generate DataFrame for topic'
    df_topic = _run(lambda: summarize_topic_df2df(topic_doc_indexs, df_word, dictionary), _info)

    _info = f'add DataFrame topic deviation'
    df_word = _run(lambda: add_df_topic_deviation(topic_N, df_word, df_topic), _info)

    return RWData(dictionary=dictionary, corpus=corpus, model=model,
                  topic_doc_indexs=topic_doc_indexs, df_word=df_word, df_topic=df_topic)


# --- MARK: 専門性 ---

def summarize_topic_df2df(topic_doc_indexs, word_topics, dictionary):
    """
    専門性に関わる：トピックごとの文書数, 非包含文書数, alpha等計算
    :param topic_doc_indexs: [[d1, d2, d3, ..], [d1, d2, d3, ..], ...]
    :param word_topics: word対topicにおけるdfを保持しているDataFrame
    :param dictionary: dictionary
    :return: DataFrame DF等
    """
    topic_N = len(topic_doc_indexs)
    word_topics = word_topics[[topic for topic in range(topic_N)]]
    # 単純なDF値
    df_topic_p = [len(texts) for texts in topic_doc_indexs]
    df_all = sum(df_topic_p)
    df_topic_n = [df_all - df_p for df_p in df_topic_p]

    # 単語毎のDF値の集計
    df_word_topics = pd.DataFrame(word_topics, index=dictionary.values()).T
    df_word_topics_p = df_word_topics.sum(axis=1)
    df_word_topics_n = df_word_topics_p.sum() - df_word_topics_p

    # (dt.DF_word_p/dt.DF_p)/(dt.DF_word_n/dt.DF_n)
    topic_generality = (df_word_topics_p / df_topic_p) / (df_word_topics_n / df_topic_n)

    return pd.DataFrame({"DF_p": df_topic_p, "DF_n": df_topic_n,
                         "DF_word_p": df_word_topics_p, "DF_word_n": df_word_topics_n,
                         "generality": topic_generality})


def _word_topic_deviation(word, topic, df_word, df_topic):
    """
    専門性：単語のトピックに対する偏りを計算
    :param word: 単語
    :param topic: トピックID
    :param df_word: 単語のDataFrame DF_topic と DF_sumが必要
    :param df_topic: トピック毎の集計DataFrame
    :return: float 偏り
    """
    # word: String, topic: int, df_word: DataFrame
    DF_word_topic = df_word.loc[word]
    DF_word_topic_p = DF_word_topic.loc[topic]  # DF_{w, topic}
    DF_word_topic_n = DF_word_topic.loc["sum"] - DF_word_topic_p  # DF_{w, topic以外}

    _df_topic = df_topic.loc[topic]
    DF_topic_p = _df_topic.DF_p  # | D_{topic} |
    DF_topic_n = _df_topic.DF_n  # | D_{topic_以外} |

    alpha = _df_topic.generality  # \alpha

    p_term = DF_word_topic_p / DF_topic_p
    n_term = DF_word_topic_n / DF_topic_n

    return p_term / (p_term + alpha * n_term)


def _cal_topic_deviation_outer(topic_N, df_word, df_topic):
    """
    DataFrame のapply用の偏り計算
    :param topic_N: トピック数
    :param df_word: 単語のDataFrame
    :param df_topic: トピック毎の集計DataFrame
    :return: do
    """

    def do(row):
        word = row.name
        return pd.Series([_word_topic_deviation(word, topic, df_word, df_topic) for topic in range(topic_N)])

    return do


def add_df_topic_deviation(topic_N, df_word, df_topic):
    _cal_topic_deviation = _cal_topic_deviation_outer(topic_N, df_word, df_topic)
    df_word_topic_deviation = df_word.apply(_cal_topic_deviation, axis=1)
    col_names = [f'{DEVIATION}{t}' for t in range(topic_N)]
    df_word[col_names] = df_word_topic_deviation
    return df_word


# --- MARK: 報酬計算 ---

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def euclid_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def euclid_sim(v1, v2):
    """ 距離から類似度変換。0.0 - 1.0の間をとるように"""
    return 1 / (1 + euclid_distance(v1, v2))


def text2vec_lda(text, model, dictionary):
    """
    文をldaに基づくベクトルに変換
    :param text: ['word', 'word', 'word', 'word']
    :param model: lda model
    :param dictionary: word dict
    :return: list ベクトル
    """
    bows = dictionary.doc2bow(text)
    probs = model.get_document_topics(bows, minimum_probability=0)
    probs = [p for i, p in probs]
    return probs


def sim_sents_lda(text1, text2, model, dictionary):
    """
    LDAによる文書類似度の計測 cos類似度
    :param text1: ['word', 'word', 'word', 'word']
    :param text2: ['word', 'word', 'word', 'word']
    :param model: lda model
    :param dictionary: word dict
    :return: float 類似度
    """
    npvec = lambda s: np.array(text2vec_lda(s, model, dictionary))
    return cos_sim(npvec(text1), npvec(text2))
