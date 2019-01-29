## A Basic PyTorch Implementation of Attentional Neural Machine Translation

This is a basic implementation of attentional neural machine translation (Bahdanau et al., 2015, Luong et al., 2015) in Pytorch 0.4.
It implements the model described in [Luong et al., 2015](https://arxiv.org/abs/1508.04025), and supports label smoothing, beam-search decoding and random sampling.
With 256-dimensional LSTM hidden size, it achieves 28.13 BLEU score on the IWSLT 2014 Germen-English dataset (Ranzato et al., 2015).

### File Structure

* `nmt.py`: contains the neural machine translation model and training/testing code.
* `vocab.py`: a script that extracts vocabulary from training data
* `util.py`: contains utility/helper functions

### Example Dataset

We provide a preprocessed version of the IWSLT 2014 German-English translation task used in (Ranzato et al., 2015) [[script]](https://github.com/harvardnlp/BSO/blob/master/data_prep/MT/prepareData.sh). To download the dataset:

```bash
wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
unzip iwslt2014_ende.zip
```

Running the script will extract a`data/` folder which contains the IWSLT 2014 dataset.
The dataset has 150K German-English training sentences. The `data/` folder contains a copy of the public release of the dataset. Files with suffix `*.wmixerprep` are pre-processed versions of the dataset from Ranzato et al., 2015, with long sentences chopped and rared words replaced by a special `<unk>` token. You could use the pre-processed training files for training/developing (or come up with your own pre-processing strategy), but for testing you have to use the **original** version of testing files, ie., `test.de-en.(de|en)`.

### Environment

The code is written in Python 3.6 using some supporting third-party libraries. We provided a conda environment to install Python 3.6 with required libraries. Simply run

```bash
conda env create -f environment.yml
```

```shell
conda env update -f=environment.yml
```

```shell
conda activate pytorch0.4
```





### Usage

Each runnable script (`nmt.py`, `vocab.py`) is annotated using `dotopt`.
Please refer to the source file for complete usage.

First, we extract a vocabulary file from the training data using the command:

```bash
python vocab.py \
    --train-src=data/train.de-en.de.wmixerprep \
    --train-tgt=data/train.de-en.en.wmixerprep \
    data/vocab.json
```

This generates a vocabulary file `data/vocab.json`. 
The script also has options to control the cutoff frequency and the size of generated vocabulary, which you may play with.

To start training and evaluation, simply run `data/train.sh`. 
After training and decoding, we call the official evaluation script `multi-bleu.perl` to compute the corpus-level BLEU score of the decoding results against the gold-standard.



# CNNDAILY

```bash
python3 vocab.py --train=data/cnn-daily/train_story_list.pickle data/cnn-daily/vocab.json
```



# DATA

> https://github.com/becxer/cnn-dailymail

| data  | volume(pair) |
| ----- | ------------ |
| train | 287227       |
| dev   | 13368        |
| test  | 11490        |

## NOTE

現状、GPU v100で、

学習：1epoch = 10000(set) = 8(batch_size) * 1250(iter) = 275(sec) = 約5分
なので、全データでは1epoch = 287227(set) = 約2時間 かかる

devでの評価は 1000(set) = 11(sec)
なので、全データでは1回の評価で13368(set) = 2.4分

decode 11490(set) = 41(min)



## TODO

- [ ] 必須
  - [ ] decodeをわかりやすいように
  - [ ] rougeでの評価
- [ ] やる
  - [ ] git ammend リファクタ。特にtrain部分でのmleとramlの共通コードをまとめる
  - [ ] tensorboard的な学習の可視化
  - [ ] モデル保存の工夫（epoch毎にとかにして、あとでdecodeを確認できるように）