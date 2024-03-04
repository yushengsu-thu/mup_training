
LLM="pythia-70m"

CUDA_VISIBLE_DEVICES=0 python3 ../code/finetune_llm.py \
    --llm "EleutherAI/"$LLM \
    --max_tokens 1024 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --batch_size 16 \
    --target_dir "../checkpoint/EleutherAI/pythia-70m"
