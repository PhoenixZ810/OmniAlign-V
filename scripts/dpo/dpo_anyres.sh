set -ex

dir="./checkpoints"
name="DPO-llavaAR4-internlm2_5-7b"

OUTPUT_DIR=${dir}/${name}

GPUS=${GPUS:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mkdir -p $(dirname "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt")
# Stage 2

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path checkpoints/llavaAR4-internlm2_5-7b-sft \
    --version internlm_2 \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --meta_path="playground/meta_json/ominialign_v_dpo.json" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --bf16 True \
    --run_name ${name} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --verbose_logging True \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt"

torchrun --nproc_per_node 8 eval_run.py \
    --data MMAlignBench\
    --model ${name} \
    --path ${OUTPUT_DIR} \
