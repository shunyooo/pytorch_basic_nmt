from sumeval.metrics.rouge import RougeCalculator

from .abstract_reward import AbstractReward
from typing import List



import numpy as np


class RougeReward(AbstractReward):

    def __init__(self, args):
        self.rouge = RougeCalculator(stopwords=True, lang="en")
        self.args = args

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        文のROUGE計算
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        return self.rouge.rouge_n(summary=hypothesis, references=reference, )

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        コーパスレベルROUGE
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        scores = [self._compute_sentence_reward(ref, hyp) for (ref, hyp) in zip(references, hypotheses)]
        return np.mean(scores)
