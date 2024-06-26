# SlimProj

from ast import mod
import copy
#import imp
from operator import is_
from pickletools import optimize
from re import sub
from xml.etree.ElementTree import TreeBuilder
from sympy import O
import torch
from torch import nn, optim, threshold
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tqdm import tqdm
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse
import os
from multiprocessing import cpu_count
import shutil
import math

from accelerate import Accelerator
import os
import argparse

from torch.distributions import Normal, kl_divergence

import re



class DatasetWrapper(IterableDataset):
    def __init__(self, max_tokens, cache_dir):
        ###
        #Tune code
        '''
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            revision="CrystalCoder_phase1_checkpoint_055500",
            trust_remote_code=True
        )
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        # pre-load the dataset --> pre-training data needed:
        ####
        '''
        self.dataset = load_dataset("Open-Orca/SlimOrca-Dedup",
                split="train",
                cache_dir=self.cache_dir
        )
        '''
        self.dataset = load_dataset("iankur/SlimPajama-1B",
                split="train",
                cache_dir=self.cache_dir
        )
        ####
        #self.train_size = int(0.9 * len(self.dataset))
        self.train_size = int(0.01 * len(self.dataset))
        self.eval_size = len(self.dataset) - self.train_size
    def __train_size__(self):
        return self.train_size
    def __eval_size__(self):
        return self.eval_size

    def __iter__(self):
        # 90%: train, 10%: test
        train_dataset = self.dataset.select(range(self.train_size))
        valid_dataset = self.dataset.select(range(self.train_size, len(self.dataset)))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #for sample in train_dataset:
        for sample in valid_dataset:
            print(sample)
            instruction_system = sample['conversations'][0]["value"]
            instruction_human = sample['conversations'][1]["value"]
            response = sample['conversations'][2]["value"]
            input_ = instruction_system + "\n" + instruction_human + "\n" + response
            tokens = self.tokenizer(input_, return_tensors='pt', max_length=self.max_tokens, padding="max_length", truncation=True).input_ids
            tokens = tokens.reshape(tokens.shape[0]*tokens.shape[1])
            yield tokens


LLM="LLM360/CrystalCoder"
VERSION="CrystalCoder_phase1_checkpoint_055500"
parser = argparse.ArgumentParser()
parser.add_argument('--llm', type=str, default="LLM360/CrystalCoder", help='llm')
parser.add_argument('--max_tokens', type=int, default=1024, help='max_length')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--target_dir', type=str, default=os.getcwd()+"/../checkpoint/CrystalCoder/CrystalCoder_phase1_checkpoint_055500", help='target_dir')
parser.add_argument('--cache_dir', type=str, default=os.getcwd()+"/../cache", help='cache_dir')
parser.add_argument('--cpus', type=int, default = cpu_count(), help='cpus')
parser.add_argument('--device', type=str, default = "cuda", help='device')
parser.add_argument('--revision', type=str, default = "CrystalCoder_phase1_checkpoint_055500", help='revision')
parser.add_argument('--distill_model_config', type=str, default = "", help='distill_model_config')
parser.add_argument('--grad_step', type=int, default=64, help='grad steps')
parser.add_argument('--reduction_factor', type=int, default=4, help='reduction_factor')


args = parser.parse_args()
dataset = DatasetWrapper(args.max_tokens, args.cache_dir)

tokenizer = dataset.tokenizer
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.cpus,
)

#print(loader)
#exit()

for i, batch in enumerate(dataset):
    if i == 10:
        break
    print(batch)


