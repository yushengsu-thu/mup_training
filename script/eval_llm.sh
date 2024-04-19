# 90% dataset: 363491

#ONly need to modify LLM:

LLM="pythia-70m"


#for lr in 0.00016  0.00032  0.00064  0.00128  1e-05  2e-05  3e-05  4e-05  5e-06  8e-05
for lr in 0.00016
do
    CUDA_VISIBLE_DEVICES=0 python3 ../code/eval_llm.py \
        --llm "EleutherAI/"$LLM \
        --max_tokens 4096 \
        --learning_rate $lr \
        --batch_size 32 \
        --checkpoint EleutherAI/$LLM/$lr/10000

        #--learning_rate $lr \
        #--weight_decay 0 \
        #--target_dir "../checkpoint/EleutherAI/"$LLM
        #--batch_size 8 \
done
