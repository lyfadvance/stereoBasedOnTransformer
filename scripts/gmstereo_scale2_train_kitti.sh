#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
NUM_GPUS=1

# kitti, this is our final model for kitti submission
CHECKPOINT_DIR=checkpoints_stereo/sceneflow-gmstereo-scale2-kitti && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--no_resume_optimizer \
--stage kitti15mix \
--lr 4e-4 \
--val_dataset kitti15 \
--batch_size 2 \
--img_height 320 \
--img_width 640 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 2000 \
--save_ckpt_freq 2000 \
--save_latest_ckpt_freq 1000 \
--num_steps 500000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log




