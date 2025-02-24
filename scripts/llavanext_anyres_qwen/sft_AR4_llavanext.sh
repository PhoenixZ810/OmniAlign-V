#!/bin/bash
set -ex

dir="./checkpoints"
name="llavaAR4-qwen2_5-32b-sft-OmniAlign"

OUTPUT_DIR=${dir}/${name}

GPUS=${GPUS:-32}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mkdir -p $(dirname "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt")

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path models/qwen/qwen2.5-32B-Instruct \
    --version qwen_2 \
    --meta_path playground/meta_json/internvl_llava_llavanext/llava_next_finetune.json \
    --vision_tower /fs-computility/mllm1/shared/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-qwen25-32b-pretrain/mm_projector.bin \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints  "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt"

torchrun --nproc_per_node 8 eval_run.py \
    --data MMVet MathVista_MINI HallusionBench MMStar OCRBench AI2D_TEST MMMU_DEV_VAL MMBench_V11 MMAlignBench\
    --model ${name} \
    --path ${OUTPUT_DIR} \