import copy

import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tqdm import tqdm
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers import AutoModel, AutoTokenizer
import argparse
import os
from multiprocessing import cpu_count
import shutil
import math

# split data, 1:9
# add ppl

PROJECT="mup"
ENTITY="mbzuai-llm"


def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    assert attention_mask is None and head_mask is None, "Not implemented"
    with cuda.sdp_kernel(enable_math=False):
        out = F.scaled_dot_product_attention(
            query.half(),
            key.half(),
            value.half(),
            is_causal=True,
        ).float()
    return out, None

# patch attention to save a lot of memory
GPTNeoXAttention._attn = _attn_wrapper


class DatasetWrapper(IterableDataset):
    def __init__(self, max_tokens, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir

        self.dataset = load_dataset("Open-Orca/SlimOrca-Dedup",
                split="train",
                cache_dir=self.cache_dir
        )
        self.train_size = int(0.9 * len(self.dataset))
        self.eval_size = len(self.dataset) - self.train_size

    def __train_size__(self):
        return self.train_size

    def __train_size__(self):
        return self.eval_size

    def __iter__(self):

        # 90%: train, 10%: test
        train_dataset = self.dataset.select(range(self.train_size))
        valid_dataset = self.dataset.select(range(self.train_size, len(self.dataset)))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #for sample in train_dataset:
        for sample in valid_dataset:
            instruction_system = sample['conversations'][0]["value"]
            instruction_human = sample['conversations'][1]["value"]
            response = sample['conversations'][2]["value"]
            input_ = instruction_system + "\n" + instruction_human + "\n" + response
            tokens = self.tokenizer(input_, return_tensors='pt', max_length=self.max_tokens, padding="max_length", truncation=True).input_ids
            tokens = tokens.reshape(tokens.shape[0]*tokens.shape[1])
            yield tokens



class Evaler:
    def __init__(self, parser):

        #self.max_tokens = 2**13
        self.llm = parser.llm
        self.max_tokens = parser.max_tokens
        self.grad = 64
        self.step = 0
        self.learning_rate = parser.learning_rate
        self.weight_decay = parser.weight_decay
        self.batch_size = parser.batch_size
        self.cache_dir = parser.cache_dir
        self.cpus = parser.cpus
        self.device = parser.device
        self.target_dir = parser.target_dir
        self.checkpoint = parser.checkpoint

        self.dataset = DatasetWrapper(self.max_tokens, self.cache_dir)

        #tensor([  50,   27,  187,  ..., 5471, 1422, 1912])
        #torch.Size([1024])

        #tensor([[1394,  403,  271,  ...,    0,    0,    0]])
        #torch.Size([1, 1024])

        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.cpus,
        )
        self.eval_max_batch = math.ceil(self.loader.dataset.eval_size/self.batch_size)
        self.train_max_batch = math.ceil(self.loader.dataset.train_size/self.batch_size)


        self.model = model = GPTNeoXForCausalLM.from_pretrained(
            self.checkpoint
            #self.llm,
            #cache_dir = self.cache_dir,
        ).to(self.device)

        self.show_params()

        self.model = self.model.eval()

    def show_params(self):
        model = self.model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.gpt_neox.embed_in.parameters())
        emb_params += list(model.embed_out.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)

    #def train_step(self, batch):
    def eval_step(self, batch):
        batch = batch.to(self.device)
        x, y = batch[:, :-1], batch[:, 1:]
        #with torch.autocast(device_type="cuda", enabled=True):
        with torch.no_grad():
            #print("=======how to self-implement training: figure it out=========")
            #refer: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf
            #refer: https://gist.github.com/NaxAlpha/3d69432aa81a9ab47dee70c7a16ad8a5

            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        #self.scaler.scale(loss / self.grad).backward()
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        return loss



    #def train(self):
    def eval(self):
        #Currently logged in as: yusheng-su (mbzuai-llm). Use `wandb login --relogin` to force relogin

        '''
        target_log = "../log/"+str(finetune_llm.py)+"/"+str(self.learning_rate)
        if os.path.isdir(target_log+"/wandb"):
            # delete dir
            shutil.rmtree(target_log+"/wandb")
        #create a new one
        os.makedirs(target_log+"/wandb")


        wandb.init(
               project=PROJECT,
               entity=ENTITY,
               #notes=socket.gethostname(),
               name="training_log",
               dir=target_log,
               job_type="fine-tuning",
               reinit=True
        )
        '''

        prog = tqdm(self.loader)
        #self.opt.zero_grad()

        total_loss = 0
        max_i = 0
        stop_batch = self.eval_max_batch

        print()
        print("lr:{}".format(self.learning_rate))
        for i, batch in enumerate(prog):
            if i == stop_batch:
                break
            self.step = i + 1
            loss = self.eval_step(batch)
            total_loss += loss.item()
            print_loss = total_loss/self.step
            prog.set_description(f"loss: {print_loss:.3f}")
            #max_i = i
            '''
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=i,
            )
            '''
        total_loss = total_loss/self.step
        #prog.set_description(f"total_loss: {total_loss:.3f}")
        print()
        print(f"total_loss: {total_loss:.3f}")
        print("========================================")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="EleutherAI/pythia-70m", help='llm')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max_length')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--target_dir', type=str, default=os.getcwd()+"/../checkpoint/pythia-70m", help='target_dir')

    parser.add_argument('--cache_dir', type=str, default=os.getcwd()+"/../cache", help='cache_dir')
    #../EleutherAI/pythia-70m/
    # config.json  generation_config.json  pytorch_model.bin
    parser.add_argument('--checkpoint', type=str, default=os.getcwd()+"/../checkpoint/", help='checkpoint')
    parser.add_argument('--cpus', type=int, default = cpu_count(), help='cpus')

    parser.add_argument('--device', type=str, default = "cuda", help='device')

    parser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    args.checkpoint = os.getcwd()+"/../checkpoint/" + args.checkpoint

    evaler = Evaler(args)
    evaler.eval()
