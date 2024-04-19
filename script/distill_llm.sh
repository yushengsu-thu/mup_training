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
    --target_dir "../checkpoint/"$LLM"/"$VERSION




