
LLM="pythia-70m"

CUDA_VISIBLE_DEVICES=3 python3 ../code/finetune_llm.py \
    --llm "EleutherAI/"$LLM \
    --max_tokens 4096 \
    --learning_rate 1.6e-4 \
    --weight_decay 0 \
    --batch_size 8 \
    --target_dir "../checkpoint/EleutherAI/pythia-70m"
