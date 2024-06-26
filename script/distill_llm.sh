#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate the virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Run the training script using accelerate launch

#Only need to modify LLM:
LLM="LLM360/CrystalCoder"
VERSION="CrystalCoder_phase1_checkpoint_055500"

#CUDA_VISIBLE_DEVICES=0
#accelerate launch ../code/distill_llm.py \

#python3 ../code/distill_llm.py \
accelerate launch ../code/distill_llm.py \
    --llm $LLM \
    --max_tokens 2048 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --batch_size 4 \
    --revision $VERSION \
    --grad_step 2 \
    --target_dir "../checkpoint/EleutherAI/"$LLM \
    --reduction_factor 4 \
    --distill_model_config "../distill-crystalcoder-config" \
    --training_config_dir "../config/default_config_fsdp.yaml" \


# grad_step default:64 --> so the actual batch_size can be seen as 1*64
