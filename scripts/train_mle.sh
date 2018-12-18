#!/bin/sh -x

train_mode="mle"
echo train: ${train_mode}

task_name="cnn-daily"

data_dir="data/"${task_name}
vocab=${data_dir}"/vocab.json"
train_file=${data_dir}"/train_story_list_10k.pickle"
dev_file=${data_dir}"/val_story_list_1k.pickle"
test_file=${data_dir}"/test_story_list.pickle"

now=`date "+%Y%m%dT%H%M%S"`
work_dir="work_dir/"${task_name}"/"${train_mode}"/"${now}

train_sample_tgt=${data_dir}"/samples.txt"

mkdir -p ${work_dir}
echo save results to ${work_dir}

train_mode_val="train_"${train_mode}
python nmt.py \
    ${train_mode_val} \
    --cuda \
    --vocab ${vocab} \
    --train ${train_file} \
    --dev ${dev_file} \
    --input-feed \
    --valid-niter 250 \
    --log-every 50 \
    --batch-size 8 \
    --dev-decode-limit 20 \
    --valid-metric 'ppl' \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --save-to ${work_dir}/model.bin \
    --log-data ${work_dir}/log_data.pickle \
    --notify-slack \
    --notify-slack-every 100 \
    --lr-decay 0.5 \
    --max-epoch 10 \
    2>${work_dir}/err.log

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_file} \
    ${work_dir}/decode.txt

#perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt >>${work_dir}/err.log
