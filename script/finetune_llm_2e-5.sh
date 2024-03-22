
LLM="pythia-70m"

CUDA_VISIBLE_DEVICES=1 python3 ../code/finetune_llm.py \
    --llm "EleutherAI/"$LLM \
    --max_tokens 4096 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --batch_size 8 \
    --target_dir "../checkpoint/EleutherAI/"$LLM
