from .bleu import BlueReward
from .deviation import DeviationReward
from .deviation_diff import DeviationDiffReward
from .shorten import ShortenReward


def load_reward_calculator(args):
    metric = args['--valid-metric']
    print(f'load_reward_calculator: {metric}')
    if metric == 'bleu':
        return BlueReward(args)
    elif metric == 'deviation':
        return DeviationReward(args)
    elif metric == 'deviation_diff':
        return DeviationDiffReward(args)
    elif metric == 'shorten':
        return ShortenReward(args)
    else:
        print(f'{metric} は実装されていません')
