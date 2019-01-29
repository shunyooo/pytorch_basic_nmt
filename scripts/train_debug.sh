#!/bin/sh -x

train_mode="mle"
echo train: ${train_mode}

task_name="de-en"

data_dir="data/"${task_name}
vocab=${data_dir}"/vocab_en.json"
train_file=${data_dir}"/train.de-en.en.wmixerprep"
dev_file=${data_dir}"/valid.de-en.en"
test_file=${data_dir}"/test.de-en.en"

now=`date "+%Y%m%dT%H%M%S"`
work_dir="work_dir/"${task_name}"/"${train_mode}"-debug/"${now}


train_sample_tgt=${data_dir}"/samples.txt"

mkdir -p ${work_dir}
echo save results to ${work_dir}
# echo less ${work_dir}/err.log | pbcopy # only Mac

train_mode_val="train_"${train_mode}
python nmt.py \
    ${train_mode_val} \
    --vocab ${vocab} \
    --train ${train_file} \
    --dev ${dev_file} \
    --input-feed \
    --valid-niter 20 \
    --log-every 1 \
    --batch-size 1 \
    --dev-decode-limit 20 \
    --valid-metric 'bleu' \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --save-to ${work_dir}/model.bin \
    --log-data ${work_dir}/log_data.pickle \
    --notify-slack \
    --lr-decay 0.5 \
    --ppl-batch-size 32 \
    --debug \
    2>${work_dir}/err.log

#python nmt.py \
#    decode \
#    --cuda \
#    --beam-size 5 \
#    --max-decoding-time-step 100 \
#    ${work_dir}/model.bin \
#    ${test_file} \
#    ${work_dir}/decode.txt
#
#perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt >>${work_dir}/err.log
