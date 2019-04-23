import json
import pickle

from .abstract_reward import AbstractReward
from typing import List, Dict, Callable


from .utils import euclid_sim
from .cons import DEVIATION

import numpy as np


class DeviationReward(AbstractReward):

    def __init__(self, args):
        # 必要なデータを読み込んでおく
        self.args = args
        self.target_vec = self.load_target_vec(args)
        self.text2vec_deviation = self.load_text2vec_deviation(args)

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        生成文が target_vec とどれだけ近いかを報酬として計測
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        return euclid_sim(self.target_vec, self.text2vec_deviation(hypothesis))

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        コーパスレベル報酬計算
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        scores = [self._compute_sentence_reward(ref, hyp) for (ref, hyp) in zip(references, hypotheses)]
        return np.mean(scores)

    @staticmethod
    def load_target_vec(args: Dict) -> [float]:
        """
        target のベクトルを読み込み、返す
        :return:
        """
        target_vec_path = args['--reward-target-vector']
        with open(target_vec_path, 'r') as f:
            target_vec = np.array(json.load(f))
            print(f'reward target vec: {target_vec}')
            return target_vec

    @staticmethod
    def load_text2vec_deviation(args: Dict) -> Callable:
        """
        text2vec_deviation を返す。
        :return:
        """
        # データ読み込み
        data_path = args['--preprocessed-data']
        with open(data_path, "rb") as f:
            preprocessed_data = pickle.load(f)
            text2vec_deviation = DeviationReward.text2vec_deviation_outer(
                preprocessed_data.df_word,
                len(preprocessed_data.df_topic),
            )
            return text2vec_deviation

    @staticmethod
    def text2vec_deviation_outer(df_word, topic_N):
        """
        文を専門性（Deviation）に基づくベクトルに変換
        高速化のため、辞書に変換し、保存
        :param df_word: 単語のDataFrame
        :param topic_N: トピック数
        :return: 計算コールバック
        """
        col_names = [f'{DEVIATION}{t}' for t in range(topic_N)]
        df_word_dict = df_word[col_names].T.to_dict()

        def do(text):
            """
            文をベクトルに変換
            :param text:  ['word', 'word', 'word', 'word']
            :return: list ベクトル
            """
            assert type(text) == list or type(text).__module__ == np.__name__
            words_count = 0
            vec = np.zeros(topic_N, dtype=float)
            for word in text:
                if word in df_word_dict.keys():
                    deviations = np.array(list(df_word_dict[word].values()))
                    vec += deviations
                    words_count += 1
            vec = vec / words_count if words_count > 0 else np.zeros(topic_N, dtype=float)
            # print(f'{" ".join(text)} | vec: {vec} | words_count: {words_count}')
            return vec

        return do

    @staticmethod
    def text2vec_deviation_legacy(text, df_word, topic_N):
        """
        文を専門性（Deviation）に基づくベクトルに変換
        WARNING: 遅い
        :param text:  ['word', 'word', 'word', 'word']
        :param df_word: 単語のDataFrame
        :param topic_N: トピック数
        :return: list ベクトル
        """
        col_names = [f'{DEVIATION}{t}' for t in range(topic_N)]
        vec = np.zeros(topic_N, dtype=float)
        words_count = 0
        for word in text:
            if word in df_word.index:
                deviations = df_word[col_names].loc[word]
                vec += deviations.values
                words_count += 1
        return vec / words_count

    @staticmethod
    def sim_sents_deviation_legacy(text1, text2, df_word, topic_N):
        """
        専門性（Deviation）による文書類似度の計測
        :param text1:
        :param text2:
        :param df_word:
        :param topic_N:
        :return:
        """
        t2v = lambda text: DeviationReward.text2vec_deviation_legacy(text, df_word, topic_N)
        return euclid_sim(t2v(text1), t2v(text2))
