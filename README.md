# mup_training
Explore the µP

## Setup
```bash
pip install -r requirment.txt
```

## Collect activation
```bash
cd script

bash gen_neurons.sh
```

## Analysis
```bash
cd script

bash analysis.sh
```

## Neuron analysis in token-level and get the results in visual
```bash
cd script

bash analysis_wordlevel.sh
```
Description:
```bash
CUDA_VISIBLE_DEVICES=1 python3 ../code/analysis_wordlevel.py \
    --num_of_samples 20
```
`--num_of_samples` is the number of statistical samples 


## Fine-tune LLms
```bash
cd script

#finetune_llm_x: x means learning rate
bash finetune_llm.sh
```

Excel: [Link](https://docs.google.com/spreadsheets/d/1ZZ0mwfliMvH0N7WlwK_gDtJPsxi_l27XxKmCstRLpAI/edit?usp=sharing)
