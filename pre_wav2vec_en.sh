export CUDA_VISIBLE_DEVICES=3
pwd=/home/data/jiangjin/TAP_ASR

dataset=LIBRISPEECH_CLEAN
python ./data/en/wav2vec_feature_extractor_en.py \
        --audio_data_dir ${pwd}/data/en/${dataset}/audio-feature/train.list \
        --f_wav2vec_path ${pwd}/data/en/${dataset}/audio-feature/wav2vec_feature.h5