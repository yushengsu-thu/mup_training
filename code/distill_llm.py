
import copy
import torch
from torch import nn, optim
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


class LargerModel:
    def __init__(self, parser):
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
        self.distill_model_config = parser.distill_model_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            revision = self.revision,
            cache_dir = self.cache_dir,
            trust_remote_code=True
        )

    def forward(self, x):
        x = x.to(self.model.device)
        z = self.model(x, output_hidden_states=True)
        return z
    

class SmallerModel:
    def __init__(self, parser):
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
        self.distill_model_config = parser.distill_model_config
        self.reduction_factor = parser.reduction_factor
        # 加载原始模型的配置并修改为新的维度
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.llm,
        #     revision = self.revision,
        #     cache_dir = self.cache_dir,
        #     trust_remote_code=True
        # )
       
        
        # self.model_config = AutoConfig.from_pretrained(self.llm, trust_remote_code=True)
        # self.model_config.rotary_dim = int(self.model_config.rotary_dim / self.reduction_factor)
        # self.model_config.n_embd = int(self.model_config.n_embd / self.reduction_factor)
        # self.model_config.n_inner = int(self.model_config.n_inner / self.reduction_factor)
        # self.model_config.hidden_size = int(self.model_config.hidden_size / self.reduction_factor)
        #model_config.num_attention_heads = int(model_config.num_attention_heads / self.reduction_factor)
        #model_config.intermediate_size = int(model_config.intermediate_size / self.reduction_factor)

        # self.model = AutoModelForCausalLM.from_config(
        #     self.model_config,
        #     trust_remote_code=True
        # ).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            revision = self.revision,
            cache_dir = self.cache_dir,
            trust_remote_code=True
        ) 

        # downsampling the weights 
        self.reduce()

    
    def reduce(self):
        # # Create a copy of the state_dict for modifications
        state_dict = self.model.state_dict()
        
        # Iterate over the state_dict and modify parameters
        # for name, param in model_original.named_parameters():
        for name, param in self.model.named_parameters():
            if param.dim() == 2:
                # 2D weight matrices
                if param.size(0) == param.size(1):
                    # Subsample and scale square matrices
                    # state_dict[name] = self._subsample_and_scale(param)
                    param.data = self._subsample_and_scale(param) 
                else:
                    # Handle rectangular matrices by subsampling only the larger dimension
                    # state_dict[name] = self._subsample_rectangular_matrices(param)
                    param.data = self._subsample_rectangular_matrices(param)
            else:
                # embedding, bias, .... (1D)
                # state_dict[name] = self._subsample_embeddings(param)
                param.data = self._subsample_embeddings(param)
        # Load the modified state_dict back to the model
        # self.model.load_state_dict(state_dict, strict=False)   
        
    def _subsample_embeddings(self, embeddings):
        #print(embeddings.shape)
        indices = torch.arange(0, embeddings.size(0), self.reduction_factor)
        subsampled_matrix = embeddings[indices]
        #print(subsampled_matrix.shape)
        return subsampled_matrix

    def _subsample_and_scale(self, matrix):
        #print(matrix.shape)
        indices = torch.arange(0, matrix.size(0), self.reduction_factor)
        subsampled_matrix = matrix[indices][:, indices] * self.reduction_factor
        #print(subsampled_matrix.shape) 
        return subsampled_matrix 

    def _subsample_rectangular_matrices(self, matrix):
        # Determine which dimension is larger
        if matrix.size(0) > matrix.size(1):
            # Subsample only along the larger dimension
            indices = torch.arange(0, matrix.size(0), self.reduction_factor)
            subsampled_matrix = matrix[indices, :]
            return subsampled_matrix 
        else:
            indices = torch.arange(0, matrix.size(1), self.reduction_factor)
            subsampled_matrix = matrix[:, indices]
            return subsampled_matrix 

    def forward(self, x):
        x = x.to(self.model.device)
        z = self.model(x, output_hidden_states=True)
        return z




class Distiller:
    def __init__(self, parser, larger_model, smaller_model):

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
        self.distill_model_config = parser.distill_model_config

        self.dataset = DatasetWrapper(self.max_tokens, self.cache_dir)

        self.larger_model = larger_model
        self.smaller_model = smaller_model
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


        #self.scaler = torch.cuda.amp.GradScaler()

        self.show_params(self.larger_model.model)
        self.show_params(self.smaller_model.model)
        
        #self.distill_model = distill_model = self.distill_model.train()
        #self.model = self.model.eval()

        '''
        self.opt = bnb.optim.Lion(
            params=self.smaller_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            optim_bits=8,
            #fused=True,
        )
        '''
        self.opt = optim.AdamW(self.smaller_model.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        #self.distill_model = torch.compile(distill_model)
        #optimizer = optim.AdamW(self.smaller_model.parameters(), lr=self.learning_rate, weight_decay=parser.weight_decay)
        # Add seduchler
        
        ###
        '''
        - device_placement (default: True): Automatically place the model and data on the right device (CPU or GPU).
        - split_batches (default: False): Automatically split batches between devices in data parallelism.
        - fp16 (default: False): Use automatic mixed precision training (floating-point 16). This is a simple way to use mixed precision without the need to configure it manually.
        - cpu (default: False): Force the use of CPU even if GPUs are available. Useful for debugging or when GPU resources are not desired.
        - deepspeed_plugin (default: None): Configuration for using DeepSpeed with Accelerator, which allows for efficient training on multiple GPUs, achieving high performance and reduced memory usage.
        - mixed_precision (default: "no"): Set the mixed precision policy. Options are "no", "fp16", and "bf16". This specifies whether to use mixed precision and which type. "fp16" and "bf16" refer to half-precision and bfloat16 precision, respectively.
        - _custom_ddp_plugin (not typically used by end users): Allows for a custom Distributed Data Parallel (DDP) plugin. This is more advanced usage for custom distributed training setups.
        - log_with (default: ["tqdm"]): Choose the libraries to use for logging progress. By default, it uses tqdm, but other loggers can be configured.
        - logging_dir (default: None): The directory to save logs to if you're using a logger that writes to files.
        - dispatch_batches (default: False): This argument is for internal use, concerning how batches are distributed across devices.
        - zero3 (default: False): Enable ZeRO Stage 3 optimization with DeepSpeed, which dramatically reduces memory usage at scale.
        - cpu_offload (default: False): Whether to offload parts of the model or computations to the CPU, usually in combination with DeepSpeed to save GPU memory.
        - gradient_accumulation_steps (default: 1): Number of steps to accumulate gradients before updating model parameters, which can be useful for effectively increasing the batch size without increasing the memory consumption.
        '''
        ###
        

        ## I want to uniformly sampling d/2 coordinates from each layers of self.larger_model and use them to initlize the self.smaller_model
        # I want to uniformly sampling d/2 coordinates from each layers of self.larger_model and use them to initlize the self.smaller_model
        '''
        d_l = self.smaller_model.model.config.hidden_size
        d = self.smaller_model.model.config.hidden_size
        print(d_l, d) 
        for layer_large, layer_small in zip(self.larger_model.model.h, self.smaller_model.model.h):
            layer_large_size = layer_large.mlp.c_proj.out_channels
            layer_small_size = layer_small.mlp.c_proj.out_channels
            indices = torch.randperm(layer_large_size)[:d//2]
            layer_small.mlp.c_proj.weight.data[:, indices] = layer_large.mlp.c_proj.weight.data[:, indices]
            print(layer_large_size, layer_small_size)
        print(111111)
        exit()
        '''
        
        
        accelerator = Accelerator(
            gradient_accumulation_steps=self.grad,
        )
        
        self.larger_model, self.smaller_model, self.opt, self.loader= accelerator.prepare(
            self.larger_model, self.smaller_model, self.opt, self.loader
        )


    def show_params(self, model):
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


        print("======Model Para=========")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.transformer.wte.parameters())
        emb_params += list(model.lm_head.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)
        print("===========================")

        
        '''
        for name, param in model.named_parameters():
            print(f"{name}: {param.size()}")
        print("===========================")
        '''


        '''
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        '''

    def loss_layer(self, output_large, output_small):
        large_hidden_states = output_large.hidden_states
        large_hidden_states = torch.stack(large_hidden_states)
        small_hidden_states = output_small.hidden_states  
        small_hidden_states = torch.stack(small_hidden_states)
        print("======")
        print(large_hidden_states.shape) #torch.Size([33, 4, 2048, 4096])
        print(small_hidden_states.shape) #torch.Size([33, 4, 2048, 1024])
        # print("======")
        # large_hidden_states = large_hidden_states.view(-1, large_hidden_states.size(-1))
        # small_hidden_states = small_hidden_states.view(-1, small_hidden_states.size(-1))
        # print(large_hidden_states.shape)
        # print(small_hidden_states.shape)
        print("======")
        print("======")
        exit()
        # calculate the loss between large_hidden_states and small_hidden_states
        loss = torch.nn.CrossEntropyLoss()(large_hidden_states, small_hidden_states.argmax(dim=-1))

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
        accumulated_loss = 0.0
         
        self.larger_model.model.train()
        self.smaller_model.model.train()
        
        print()
        print("lr:{}".format(self.learning_rate))
        for i, batch in enumerate(prog):
            if i == stop_batch:
                break
            self.step = i + 1
            output_large = self.larger_model.forward(batch)
            output_small = self.smaller_model.forward(batch)
            loss = self.loss_layer(output_large, output_small) 
            #model_z = self.inference_step(batch)
            #distill_model_loss = self.train_step(batch, model_z)
            #total_loss += distill_model_loss.item()
            accumulated_loss += loss.item()
            total_loss += loss.item()
            #total_loss += loss.item()
            #print_loss = total_loss/self.step
            #prog.set_description(f"loss: {print_loss:.3f}")
            prog.set_description(f"loss: {loss.item():.3f}")
            #max_i = i
            '''
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=i,
            )
            '''
            # # 检查参数是否变化
            # for initial_param, trained_param in zip(initial_teacher_params, teacher_model.parameters()):
            #     if not torch.equal(initial_param, trained_param):
            #         print("Parameter changed!")
            #     else:
            #         print("Parameter unchanged.")

            self.opt.zero_grad()
            self.accelerator.backward(loss)
            
            
            if (i + 1) % self.grad == 0:
                self.opt.step()
                prog.set_description(f"accumulated_loss: {total_loss:.3f}")
                accumulated_loss = 0.0
                
                # self.scaler.step(self.opt)
                # self.scaler.update()
                # self.opt.zero_grad()
        total_loss = total_loss/self.step
        #prog.set_description(f"total_loss: {total_loss:.3f}")
        print()
        print(f"total_loss: {total_loss:.3f}")
        print("========================================")



def main():
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

    parser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    args = parser.parse_args()
    #args.checkpoint = os.getcwd()+"/../checkpoint/" + args.checkpoint

    smaller_model = SmallerModel(args)

    larger_model = LargerModel(args)

    distiller = Distiller(args, larger_model, smaller_model)
    distiller.distill()

    
if __name__ == "__main__":  
    main()