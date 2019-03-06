from .abstract_reward import AbstractReward
from typing import List



import numpy as np


class ShortenReward(AbstractReward):

    def __init__(self, args):
        self.args = args

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        return self.sent_length(reference) - self.sent_length(hypothesis)

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        コーパスレベルROUGE
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        scores = [self._compute_sentence_reward(ref, hyp) for (ref, hyp) in zip(references, hypotheses)]
        return np.mean(scores)

    @staticmethod
    def sent_length(text):
        return len(' '.join(text))
