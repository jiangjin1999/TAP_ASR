# Cross Modal Training ASR Error Correction With Contrastive Learning

This the office code for the paper [Cross Modal Training ASR Error Correction With Contrastive Learning](https://arxiv.org/abs/).


# Environment
- Python: 3.8
- Cuda: 10.2
- Packages: 
  ```shell
  pip install -r requirements.txt
  ```

# Data Preperation

## ASR Data

AISHELL-1: https://openslr.org/33

AIDATATANG: https://openslr.org/62 

MAGICDATA: https://openslr.org/68 

LibriSpeech-clean & LibriSpeech-other: https://openslr.org/12

Use Wenet to process the ASR Data(https://github.com/wenet-e2e/wenet/tree/main/examples)



## Data Preprocess
From *.wav to *.h5

Only process "train" part data in "train/test/dev" split.

### wav2vec model preparation
For en:
facebook/wav2vec2-base-960h: https://huggingface.co/facebook/wav2vec2-base-960h

Local: `TAP_ASR/pretrained-model/wav2vec_pretrain_model/en/`

For zh:
ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt: https://huggingface.co/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt

Local: `TAP_ASR/pretrained-model/wav2vec_pretrain_model/zh/`

### wav2vec feature extractor
(This step need large "Hard drive capacity" >= 100GB)

For en:
`TAP_ASR/data/en/wav2vec_feature_extractor_en.py`

For zh:
`TAP_ASR/data/zh/wav2vec_feature_extractor_zh.py`



# Pretrained-model Preparation

## en
BART: bart-base

Huggingface: https://huggingface.co/facebook/bart-base

Local: `TAP_ASR/pretrained-model/en/BART/`

## zh
BART: bart-base

Huggingface: https://huggingface.co/fnlp/bart-base-chinese

Local: `TAP_ASR/pretrained-model/zh/BART/`

Phonetic Encoder: url...

# Train and Test 





# Other


## dataset statistics