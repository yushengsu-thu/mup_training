import copy

import torch
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


# split data, 1:9
# add ppl

PROJECT="mup"
ENTITY="mbzuai-llm"


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
        self.dataset = load_dataset("Open-Orca/SlimOrca-Dedup",
                split="train",
                cache_dir=self.cache_dir
        )
        self.train_size = int(0.9 * len(self.dataset))
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
            instruction_system = sample['conversations'][0]["value"]
            instruction_human = sample['conversations'][1]["value"]
            response = sample['conversations'][2]["value"]
            input_ = instruction_system + "\n" + instruction_human + "\n" + response
            tokens = self.tokenizer(input_, return_tensors='pt', max_length=self.max_tokens, padding="max_length", truncation=True).input_ids
            tokens = tokens.reshape(tokens.shape[0]*tokens.shape[1])
            yield tokens



class Distiller:
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
        self.revision = parser.revision
        self.dataset = DatasetWrapper(self.max_tokens, self.cache_dir)
        self.distill_model_config = parser.distill_model_config
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
        self.eval_max_batch = math.ceil(self.loader.dataset.__eval_size__()/self.batch_size)
        self.train_max_batch = math.ceil(self.loader.dataset.__train_size__()/self.batch_size)



        #config = AutoConfig.from_pretrained('LLM360/CrystalCoder', trust_remote_code=True)
        #config.save_pretrained('../distill-crystalcoder-config')


        self.scaler = torch.cuda.amp.GradScaler()
        config = AutoConfig.from_pretrained(self.distill_model_config, trust_remote_code=True)
        self.distill_model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True
        ).to(self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            revision = self.revision,
            cache_dir = self.cache_dir,
            trust_remote_code=True
        ).to(self.device)

        self.show_params()

        self.distill_model = self.distill_model.train()
        self.model = self.model.eval()


    def show_params(self):
        '''
        CrystalCoderLMHeadModel(
        (transformer): CrystalCoderModel(
            (wte): Embedding(32032, 4096)
            (drop): Dropout(p=0.0, inplace=False)
            (h): ModuleList(
            (0-31): 32 x CrystalCoderBlock(
                (ln_1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
                (attn): CrystalCoderAttention(
                (c_attn): Conv1D()
                (c_proj): Conv1D()
                (attn_dropout): Dropout(p=0.0, inplace=False)
                (resid_dropout): Dropout(p=0.0, inplace=False)
                )
                (ln_2): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
                (mlp): CrystalCoderMLP(
                (c_fc): Conv1D()
                (c_fc2): Conv1D()
                (c_proj): Conv1D()
                (act): SwiGLUActivation()
                (dropout): Dropout(p=0.0, inplace=False)
                )
            )
            )
            (ln_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (lm_head): Linear(in_features=4096, out_features=32032, bias=False)
        )
        '''
        print("======Larger Model=========")
        model = self.model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.transformer.wte.parameters())
        emb_params += list(model.lm_head.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)
        print("======Distilled Model=========")
        model = self.distill_model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.transformer.wte.parameters())
        emb_params += list(model.lm_head.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)
        print("==============================")


    #def train_step(self, batch):
    def eval_step(self, batch):
        batch = batch.to(self.device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.no_grad():
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        return loss


    def train_step(self, batch):
        batch = batch.to(self.device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", enabled=True):
            #print("=======how to self-implement training: figure it out=========")
            #refer: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf
            #refer: https://gist.github.com/NaxAlpha/3d69432aa81a9ab47dee70c7a16ad8a5
            z = self.distill_model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        return loss

        '''
        batch = batch.to(self.device)
        print("=======")
        print(batch.size)
        print(len(batch))
        print("=======")
        x, y = batch[:, :-1], batch[:, 1:]
        print(x.shape)
        print("-----")
        print(y.shape)
        print("-----")
        exit()
        with torch.autocast(device_type="cuda", enabled=True):
            z = self.model(x).logits
            y = y.reshape(-1)
            #print(z.shape) #torch.Size([1, 1023, 50304])
            #print(y.shape) #torch.Size([1023])
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        return loss
        '''



    #def train(self):
    def distill(self):
        #Currently logged in as: yusheng-su (mbzuai-llm). Use `wandb login --relogin` to force relogin

        '''
        target_log = "../log/"+str(self.target_dir)+"/"+str(self.learning_rate)
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
        stop_batch = self.train_max_batch

        print()
        print("lr:{}".format(self.learning_rate))
        for i, batch in enumerate(prog):
            if i == stop_batch:
                break
            self.step = i + 1
            model_loss = self.eval_step(batch)
            distill_model_loss = self.train_step(batch)
            total_loss += distill_model_loss.item()
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
            if (i + 1) % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
        total_loss = total_loss/self.step
        #prog.set_description(f"total_loss: {total_loss:.3f}")
        print()
        print(f"total_loss: {total_loss:.3f}")
        print("========================================")




if __name__ == "__main__":
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

    parser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    #args.checkpoint = os.getcwd()+"/../checkpoint/" + args.checkpoint

    distiller = Distiller(args)
    distiller.distill()

    
    