from typing import List


class AbstractReward(object):
    """Reward"""

    def __init__(self, args):
        raise NotImplementedError()

    def compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        単一文書レベルで報酬を計算する。前処理付き。
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        reference, hypothesis = self.preprocess_sentence(reference, hypothesis)
        return self._compute_sentence_reward(reference, hypothesis)

    def compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        コーパスレベルで報酬を計算する。前処理付き。
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        references, hypotheses = self.preprocess_corpus(references, hypotheses)
        return self._compute_corpus_reward(references, hypotheses)

    def _compute_sentence_reward(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        プライベート：単一文書レベルで報酬を計算する
        :param reference: 参照文
        :param hypothesis: 生成文
        :return: 報酬
        """
        # MARK:要実装。抽象メソッド
        raise NotImplementedError()

    def _compute_corpus_reward(self, references: List[List[str]], hypotheses: List[List[str]]) -> float:
        """
        プライベート：コーパスレベルで報酬を計算する
        :param references: 参照文リスト
        :param hypotheses: 生成文リスト
        :return: 報酬
        """
        # MARK:要実装。抽象メソッド
        raise NotImplementedError()

    def preprocess_sentence(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        文の前処理
        :param reference:
        :param hypothesis:
        :return:
        """
        if reference[0] == '<s>':
            reference = reference[1:-1]

        return reference, hypothesis

    def preprocess_corpus(self, references: List[List[str]], hypotheses: List[List[str]]) -> (
            List[List[str]], List[List[str]]):
        """
        コーパスの前処理
        :param references:
        :param hypotheses:
        :return:
        """
        if references[0][0] == '<s>':
            references = [ref[1:-1] for ref in references]

        return references, hypotheses
