# 代码中已经全部设置为False,设置为True时，需要解除 注释
# export CUDA_VISIBLE_DEVICES=1,2,3
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=22346 model-trainer.py \
#         --is_use_DDP \
#         --max_seq_length 75 \
#         --current_dataset MAGICDATA \
#         --batch_size 80
#         # --is_audio \
#         # --is_phoneme \
#         # --is_jointly_train  \
#         # --is_CL_train \
#         # --is_jointly_train_zero
#         # --is_limited_CL_train
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=22345 model-trainer.py \
        --is_use_DDP \
        --current_dataset AIDATATANG \
        --batch_size 40 \
        --is_phoneme \
        --is_audio
        # --is_jointly_train \
        # --is_CL_train
        # --is_jointly_train_zero
       # --is_limited_CL_train
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=2234 model-trainer.py \
#         --is_use_DDP \
#         --current_dataset AISHELL-1 \
#         --batch_size 30 \
#         --is_phoneme \
#         --is_audio \
#         --is_jointly_train \
#         --is_CL_train \
#         # --is_limited_CL_train

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=2234 model-trainer.py \
#         --is_use_DDP \
#         --current_dataset AISHELL-1 \
#         --batch_size 50 \
#         --is_audio \
#         --is_jointly_train
#         # --is_CL_train \
#         # --is_limited_CL_train
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=22345 model-trainer.py \
#         --is_use_DDP \
#         --current_dataset AIDATATANG \
#         --batch_size 30 \
#         --is_audio \
#         --is_jointly_train \
#         --is_CL_train
#         # --is_limited_CL_train

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=22345 model-trainer.py \
#         --is_use_DDP \
#         --current_dataset AISHELL-1 \
#         --batch_size 30 \
#         --is_audio \
#         --is_phoneme \
#         --is_jointly_train \
#         --is_CL_train \
#         --is_limited_CL_train
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=22345 model-trainer.py \
#         --is_use_DDP \
#         --current_dataset AISHELL-1 \
#         --batch_size 25 \
#         --is_audio \
#         --is_phoneme \
#         --is_jointly_train \
#         --is_CL_train \
#         --is_limited_CL_train \
#         --is_jointly_train_zero