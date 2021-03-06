#!/bin/sh

train_mode="$1"
echo train: ${train_mode}

vocab="data/vocab.json"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

now=`date "+%Y%m%dT%H%M%S"`
work_dir="work_dir/"${train_mode}"/"${now}

train_sample_tgt="data/samples.txt"

mkdir -p ${work_dir}
echo save results to ${work_dir}

train_mode_val="train_"${train_mode}
python nmt.py \
    ${train_mode_val} \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --input-feed \
    --valid-niter 2400 \
    --batch-size 64 \
    --dev-decode-limit 2000 \
    --valid-metric 'blue' \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --save-to ${work_dir}/model.bin \
    --log-data ${work_dir}/log_data.pickle \
    --notify-slack \
    --raml-sample-file ${train_sample_tgt} \
    --lr-decay 0.5 2>${work_dir}/err.log

echo decode

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt >>${work_dir}/err.log
