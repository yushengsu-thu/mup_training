# 90% dataset: 363491

#Only need to modify LLM:

LLM="LLM360/CrystalCoder"
VERSION="CrystalCoder_phase1_checkpoint_055500"


CUDA_VISIBLE_DEVICES=0 python3 ../code/distill_llm.py \
    --llm $LLM \
    --max_tokens 2048 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --batch_size 1 \
    --revision $VERSION \
    --grad_step 64 \
    --target_dir "../checkpoint/EleutherAI/"$LLM \
    --distill_model_config "../distill-crystalcoder-config"


# grad_step default:64 --> so the actual batch_size can be seen as 1*64
