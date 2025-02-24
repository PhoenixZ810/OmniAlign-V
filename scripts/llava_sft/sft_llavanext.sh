#!/bin/bash
set -ex

dir="./checkpoints"
name="llava-internlm2_5-7b-sft-OmniAlignV"

OUTPUT_DIR=${dir}/${name}

mkdir -p $(dirname "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt")

deepspeed --num_gpus=8 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /fs-computility/mllm1/zhaoxiangyu/code_new/LLaVA/models/internlm/internlm2_5-7b-chat \
    --version internlm_2 \
    --meta_path playground/meta_json/omnialign_v.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter checkpoints/llava-internlm2_5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt"

torchrun --nproc_per_node 8 eval_run.py \
    --data MMVet MathVista_MINI HallusionBench MMStar OCRBench AI2D_TEST MMMU_DEV_VAL MMBench_V11 MMBench_CN_V11 MMAlignBench\
    --model ${name} \
    --path ${OUTPUT_DIR} \