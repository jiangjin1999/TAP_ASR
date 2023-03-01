export CUDA_VISIBLE_DEVICES=1
pwd=/home/data/jiangjin/TAP_ASR

dataset=MAGICDATA
python ./data/zh/wav2vec_feature_extractor_zh.py \
        --audio_data_dir ${pwd}/data/zh/${dataset}/audio-feature/train.list \
        --f_wav2vec_path ${pwd}/data/zh/${dataset}/audio-feature/wav2vec_feature.h5