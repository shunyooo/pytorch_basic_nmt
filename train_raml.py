import math
import sys
import time

import numpy as np
from typing import Dict

import torch

from nmt import NMT, evaluate_ppl, evaluate_valid_metric
from utils import batch_iter, read_corpus_de_en, notify_slack_if_need, log_decode_to_tensorboard_raml, list_dict_update, \
    read_raml_train_data
from vocab import Vocab

from tensorboardX import SummaryWriter

writer = SummaryWriter(comment='NMT')
DECODE_LOG_INDEXES = [0, 10, 13, 15]

def train_raml(args: Dict):
    train_data_src = read_corpus_de_en(args['--train'], source='src')
    train_data_tgt = read_corpus_de_en(args['--train'], source='tgt')

    dev_data_src = read_corpus_de_en(args['--dev'], source='src')
    dev_data_tgt = read_corpus_de_en(args['--dev'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    ppl_batch_size = int(args['--ppl-batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    notify_slack_every = int(args['--notify-slack-every'])
    model_save_path = args['--save-to']
    dev_decode_limit = int(args['--dev-decode-limit'])
    is_debug = bool(args['--debug'])

    assert max(DECODE_LOG_INDEXES) < dev_decode_limit < len(dev_data), 'DECODEのログindexか, 数指定が不正です'


    vocab = Vocab.load(args['--vocab'])

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=args['--input-feed'],
                label_smoothing=float(args['--label-smoothing']),
                vocab=vocab)
    model.train()

    # ▼▼▼▼ RAML ▼▼▼▼
    tau = float(args['--raml-temp'])
    raml_sample_mode = args['--raml-sample-mode']
    raml_sample_size = int(args['--raml-sample-size'])
    # ▲▲▲▲ RAML ▲▲▲▲

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    # ▶︎▶︎ RAML ◀︎◀︎
    report_weighted_loss = cum_weighted_loss = 0

    # ▼▼▼▼ RAML ▼▼▼▼
    # NOTE: RAML サンプリングの読み込み or 生成 begin
    if raml_sample_mode == 'pre_sample':
        # dict of (src, [tgt: (sent, prob)])
        print('read in raml training data...', file=sys.stderr, end='')
        begin_time = time.time()
        raml_samples = read_raml_train_data(args['--raml-sample-file'], temp=tau)
        print('done[%d s].' % (time.time() - begin_time))
    else:
        raise Exception(f'sampling:{raml_sample_mode} は、まだ未実装です')
    # ▲▲▲▲ RAML ▲▲▲▲

    # log用, あとで学習の収束とか見るよう
    log_data = {'args': args}

    # slack
    _info = f"""
        begin RAML training
        ・学習：{len(train_data)}ペア
        ・テスト：{len(dev_data)}ペア, {valid_niter}iter毎
        ・バッチサイズ：{train_batch_size}
        ・1epoch = {len(train_data)}ペア = {int(len(train_data)/train_batch_size)}iter
        ・max epoch：{args['--max-epoch']}
    """
    print(_info)
    print(_info, file=sys.stderr)

    notify_slack_if_need(f"""
    {_info}
    {args}
    """, args)

    # log decode tensorboard
    log_decode_to_tensorboard_raml(-1, DECODE_LOG_INDEXES, writer, args=args, dev_data=dev_data)

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            if train_iter == 1:
                _info = f'原文例：{src_sents[0]}\n参照例：{tgt_sents[0]}'
                print(_info)
                print(_info, file=sys.stderr)

            # ▼▼▼▼ RAML ▼▼▼▼
            # src_sents 内 sent に紐づくサンプリングを取得　→　学習データとする
            raml_src_sents = []
            raml_tgt_sents = []
            raml_tgt_weights = []
            if raml_sample_mode == 'pre_sample':
                for src_sent in src_sents:
                    sent = ' '.join(src_sent)
                    tgt_samples_all = raml_samples[sent]
                    # random choice from candidate samples
                    if raml_sample_size >= len(tgt_samples_all):
                        tgt_samples = tgt_samples_all
                    else:
                        tgt_samples_id = np.random.choice(range(1, len(tgt_samples_all)),
                                                          size=raml_sample_size - 1, replace=False)
                        # [ground truth y*] + samples
                        tgt_samples = [tgt_samples_all[0]] + [tgt_samples_all[i] for i in tgt_samples_id]

                    raml_src_sents.extend([src_sent] * len(tgt_samples))
                    raml_tgt_sents.extend([['<s>'] + sent.split(' ') + ['</s>'] for sent, weight in tgt_samples])
                    raml_tgt_weights.extend([weight for sent, weight in tgt_samples])
            else:
                raise Exception(f'sampling:{raml_sample_mode} は、まだ未実装です')
            # ▲▲▲▲ RAML ▲▲▲▲

            optimizer.zero_grad()

            # ▼▼▼▼ RAML ▼▼▼▼
            weights_var = torch.tensor(raml_tgt_weights, dtype=torch.float, device=device)
            batch_size = len(raml_src_sents)
            # (batch_size)
            unweighted_loss = -model(raml_src_sents, raml_tgt_sents)
            batch_loss = weighted_loss = (unweighted_loss * weights_var).sum()
            loss = batch_loss / batch_size
            writer.add_scalar('loss/train_unweighted_loss', unweighted_loss.sum() / batch_size, train_iter)
            writer.add_scalar('loss/train_weighted_loss', loss, train_iter)
            # ▲▲▲▲ RAML ▲▲▲▲

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

            optimizer.step()

            # ▼▼▼▼ RAML ▼▼▼▼
            # only RAML: weighted_loss_val, report_weighted_loss, cum_weighted_loss
            # share MLE: batch_losses_val, report_loss, cum_loss, cum_loss
            weighted_loss_val = weighted_loss.item()
            batch_losses_val = unweighted_loss.sum().item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_weighted_loss += weighted_loss_val
            cum_weighted_loss += weighted_loss_val
            # ▲▲▲▲ RAML ▲▲▲▲

            tgt_words_num_to_predict = sum(len(s[1:]) for s in raml_src_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0 or train_iter % notify_slack_every == 0:
                _report = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                          'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                             report_weighted_loss / report_examples,
                                                                                             math.exp(
                                                                                                 report_loss / report_tgt_words),
                                                                                             cum_examples,
                                                                                             report_tgt_words / (
                                                                                                     time.time() - train_time),
                                                                                             time.time() - begin_time)
                print(_report)
                print(_report, file=sys.stderr)

                list_dict_update(log_data, {
                    'epoch': epoch,
                    'train_iter': train_iter,
                    'loss': report_loss / report_examples,
                    'ppl': math.exp(report_loss / report_tgt_words),
                    'examples': cum_examples,
                    'speed': report_tgt_words / (time.time() - train_time),
                    'elapsed': time.time() - begin_time
                }, 'train')

                train_time = time.time()
                report_loss = report_weighted_loss = report_tgt_words = report_examples = 0.

                if train_iter % notify_slack_every == 0:
                    notify_slack_if_need(_report, args)

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_weighted_loss / cum_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cum_tgt_words),
                                                                                             cum_examples), file=sys.stderr)

                cum_loss = cum_weighted_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')
                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                _begin_time = time.time()
                dev_ppl = dev_loss = 0
                # dev_ppl, dev_loss = evaluate_ppl(model, dev_data, batch_size=16)  # dev batch size can be a bit larger
                valid_metric, eval_info = evaluate_valid_metric(model, dev_data, dev_ppl, args)
                _elapsed = time.time() - _begin_time

                _report = 'validation: iter %d, dev. ppl %f, dev. %s %f , time elapsed %.2f sec' % (
                    train_iter, dev_ppl, args['--valid-metric'], valid_metric, _elapsed
                )
                print(_report)
                print(_report, file=sys.stderr)
                notify_slack_if_need(_report, args)

                if 'dev_data' in log_data:
                    log_data['dev_data'] = dev_data[:int(args['--dev-decode-limit'])]

                list_dict_update(log_data, {
                    'epoch': epoch,
                    'train_iter': train_iter,
                    'loss': dev_loss,
                    'ppl': dev_ppl,
                    args['--valid-metric']: valid_metric,
                    **eval_info,
                }, 'validation', is_save=True)

                # log to tensorboard
                writer.add_scalar('metric/dev_ppl', dev_ppl, train_iter)
                writer.add_scalar('metric/' + args['--valid-metric'], valid_metric, train_iter)
                if 'top_hyps' in eval_info:
                    log_decode_to_tensorboard_raml(train_iter, DECODE_LOG_INDEXES, writer, args=args, eval_info=eval_info)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            _report = 'early stop!'
                            notify_slack_if_need(_report, args)
                            print(_report, file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    _report = 'reached maximum number of epochs!'
                    notify_slack_if_need(_report, args)
                    print(_report, file=sys.stderr)
                    exit(0)

