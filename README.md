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

Follow this Tutorial to get the AISHELL-1 wav file: https://wenet.org.cn/wenet/tutorial_aishell.html
  - Only do the --stage -1 --stop_stage 3 to get the file: data/train/data.list
  - the format of this file is: 
    {"key": "BAC009S0002W0122", "wav": "/home/***/wenet/examples/aishell/s0/export/data/asr-data/OpenSLR/33//data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购。"}

Directly tar the train data (https://openslr.org/33), get the wav file and make the same format file also work

Put this file in path (AISHELL-1): ./TAP_ASR/data/zh/AISHELL-1/audio-feature/data.list

## Data Preprocess
From *.wav to *.h5

Only process "train" part data in "train/test/dev" split.

### wav2vec model preparation
Download the pretrained wav2vec model from huggingface

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

Get the h5 file : ./TAP_ASR/data/zh/AIDATATANG/audio-feature/wav2vec_feature.h5


# Pretrained-model Preparation

## en
BART: bart-base

Huggingface: https://huggingface.co/facebook/bart-base

Local: `TAP_ASR/pretrained-model/en/BART/`

(No pre-trained phoneme encoder for en)

## zh
BART: bart-base

Huggingface: https://huggingface.co/fnlp/bart-base-chinese

Local: `TAP_ASR/pretrained-model/zh/BART/`

Pinyin Encoder: url...

# Train and Test 


```shell
bash T_model.sh
bash TA_model.sh
bash TP_model.sh
bash TAP_model.sh
bash TA_CL_model.sh
bash TP_CL_model.sh
bash TAP_CL_model.sh
```





# Other


## dataset statistics