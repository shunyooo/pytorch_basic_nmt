from .abstract_reward import AbstractReward
from typing import List



import numpy as np

from .deviation import DeviationReward


class DeviationDiffReward(AbstractReward):

    def __init__(self, args):
        # 必要なデータを読み込んでおく
        self.args = args
        self.target_vec = DeviationReward.load_target_vec(args)
        self.text2vec_deviation = DeviationReward.load_text2vec_deviation(args)
        self.target_index = self.target_vec.argmax()

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        参照文（入力文）から生成した文がどれだけ target_vec に近づいたかを測る。
        :param reference: 参照文：入力文とみなす
        :param hypothesis: 生成文
        :return: 報酬
        """

        diff_vec = self.text2vec_deviation(hypothesis) - self.text2vec_deviation(reference)
        # target_index でどれだけ+に変化したか
        diff_value = diff_vec[self.target_index]
        return diff_value

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        デコード結果と参照=入力の、目的のインデックスにおける差分をコーパスレベルで測る。
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        scores = [self._compute_sentence_reward(ref, hyp) for (ref, hyp) in zip(references, hypotheses)]
        return np.mean(scores)
