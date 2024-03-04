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

# split data, 1:9
# add ppl


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

    def __iter__(self):

        dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train", cache_dir=self.cache_dir)
        train_size = int(0.9 * len(dataset))
        # 90%: train, 10%: test
        train_dataset = dataset.select(range(train_size))
        valid_dataset = dataset.select(range(train_size, len(dataset)))

        for sample in train_dataset:
            input_ = self.tokenizer(sample['conversations'][0]["value"]+"\n"+sample['conversations'][1]["value"])["input_ids"]
            output_ = self.tokenizer(sample['conversations'][2]["value"])["input_ids"]

            if len(input_) >= self.max_tokens:
                input_ = input_[:self.max_tokens]
                input_ = torch.tensor(input_)
            else:
                input_ = torch.tensor( [self.tokenizer.eos_token_id] * (self.max_tokens - len(input_)) + input_)

            if len(output_) >= self.max_tokens:
                output_ = output_[:self.max_tokens]
                output_ = torch.tensor(output_)
            else:
                output_ = torch.tensor([self.tokenizer.eos_token_id] * (self.max_tokens - len(output_)) + output_)

            yield {"input": input_, "output": output_}



class Trainer:
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


        self.dataset = DatasetWrapper(self.max_tokens, self.cache_dir)

        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.cpus,
        )


        self.scaler = torch.cuda.amp.GradScaler()
        self.model = model = GPTNeoXForCausalLM.from_pretrained(
            self.llm,
            #cache = self.cache_dir,
        ).to(self.device)


        self.show_params()

        self.opt = bnb.optim.Lion(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            optim_bits=8,
            # fused=True,
        )
        self.model = torch.compile(model)

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

    def train_step(self, batch):
        #batch = batch.cuda()
        x = batch['input'].to(self.device)
        y = batch['output'].to(self.device)
        #batch = batch.to(self.device)
        #x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", enabled=True):
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        return loss

    def train(self):
        #Currently logged in as: yusheng-su (mbzuai-llm). Use `wandb login --relogin` to force relogin

        wandb.init(
               project="mup",
               entity="mbzuai-llm",
               #notes=socket.gethostname(),
               name="training_log",
               dir="../",
               job_type="fine-tuning",
               reinit=True)


        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):

            self.step = i + 1

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=i,
            )
            print(1111111)
            exit()

            if (i + 1) % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            '''
            if i % 1000 == 0:
                temp_model = copy.deepcopy(self.model).half()
                temp_model.save_pretrained(
                    "<your-hf-repo-id>",
                    push_to_hub=True,
                    max_shard_size="500MB",
                )
                del temp_model
                torch.cuda.empty_cache()
            '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="EleutherAI/pythia-70m", help='llm')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max_length')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--target_dir', type=str, default="../checkpoint/pythia-70m", help='target_dir')

    parser.add_argument('--cache_dir', type=str, default=os.getcwd()+"/../cache", help='target_dir')
    parser.add_argument('--cpus', type=int, default = cpu_count(), help='cpus')

    parser.add_argument('--device', type=str, default = "cuda", help='device')

    parser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()



    trainer = Trainer(args)
    trainer.train()
