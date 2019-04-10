#!/bin/sh

train_src="data/de-en/train.de-en.en.wmixerprep"
train_tgt="data/de-en/train.de-en.en.wmixerprep"

reward="shorten"
mode="sample_ngram_word2vec"
target_vec="data/de-en/target_vec.json"
sample_output="data/de-en/"${mode}"_"${reward}".txt"
vocab="data/de-en/vocab_en.json"

preprocessed_data="data/de-en/models/lda5_153326doc_22625word_rwdata.pickle"


python process_samples.py \
    --mode ${mode} \
    --reward ${reward} \
    --vocab ${vocab} \
    --src ${train_src} \
    --tgt ${train_tgt} \
    --output ${sample_output} \
    --preprocessed_data ${preprocessed_data} \
    --sample_size 100 \
    --target_vec ${target_vec} \
    --smooth_bleu \
    --is_debug