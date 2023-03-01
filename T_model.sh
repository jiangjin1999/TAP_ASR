# 代码中已经全部设置为False,设置为True时，需要解除 注释
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2234 model-trainer.py \
        --is_use_DDP \
        --current_dataset AISHELL-1 \
        --batch_size 20
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=22345 model-trainer.py \
        --is_use_DDP \
        --current_dataset AIDATATANG \
        --batch_size 40
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=22346 model-trainer.py \
        --is_use_DDP \
        --current_dataset MAGICDATA \
        --batch_size 80
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=22346 model-trainer.py \
        --is_use_DDP \
        --current_dataset LIBRISPEECH_CLEAN \
        --batch_size 80
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=22346 model-trainer.py \
        --is_use_DDP \
        --current_dataset LIBRISPEECH_OTHER \
        --batch_size 80