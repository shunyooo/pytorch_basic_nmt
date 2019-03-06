from .abstract_reward import AbstractReward
from typing import List
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction



class BlueReward(AbstractReward):

    def __init__(self, args):
        self.args = args
        self.sm_func = self.get_smooth_func(args)

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        文のBLUE計算
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        return sentence_bleu([reference], hypothesis, smoothing_function=self.sm_func)

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        コーパスレベルBLUE
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        return corpus_bleu([[ref] for ref in references], hypotheses, smoothing_function=self.sm_func)

    @staticmethod
    def get_smooth_func(args):
        key = '--smooth-bleu'
        if key in args and args[key]:
            return SmoothingFunction().method3
