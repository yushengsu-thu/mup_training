
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

from torch.distributions import Normal, kl_divergence

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



# 1 D
def _subsample_embeddings(matrix_original, matrix_target, reduction_factor):
    #print(embeddings.shape)
    indices = torch.arange(0, matrix_original.size(0), reduction_factor)
    out_dim = int(indices.shape[0])
    target_d0 = int(matrix_target.shape[0])
    if out_dim == target_d0:
        pass
    else:
        indices = indices[:target_d0]
    subsampled_matrix = matrix_original[indices]
    return subsampled_matrix.data
def _subsample_and_scale(matrix_original, matrix_target, reduction_factor):
    #print(matrix.shape)
    indices = torch.arange(0, matrix_original.size(0), reduction_factor)
    #subsampled_matrix = matrix_original[indices][:, indices] * self.reduction_factor
    out_dim = int(indices.shape[0])
    target_d0, target_d1 = int(matrix_target.shape[0]), int(matrix_target.shape[1])
    if out_dim == target_d0:
        pass
    else:
        indices = indices[:target_d0]
    if out_dim == target_d1:
        pass
    else:
        indices = indices[:target_d1]
    subsampled_matrix = matrix_original[indices, :][:, indices] * reduction_factor
    return subsampled_matrix.data
def _subsample_embeddings_dim(matrix_original, matrix_target, reduction_factor):
    # Determine which dimension is larger
    #if matrix.size(0) < matrix.size(1):
    # Subsample only along the larger dimension
    indices_0 = torch.arange(0, matrix_original.size(0), reduction_factor)
    indices_1 = torch.arange(0, matrix_original.size(1), reduction_factor)
    out_dim_0 = int(indices_0.shape[0])
    out_dim_1 = int(indices_1.shape[0])
    target_d0, target_d1 = int(matrix_target.shape[0]), int(matrix_target.shape[1])
    if out_dim_0 == target_d0:
        pass
    else:
        indices_0 = indices_0[:target_d0]
    if out_dim_1 == target_d1:
        pass
    else:
        indices_1 = indices_1[:target_d1]
    subsampled_matrix = matrix_original[indices_0, :][: ,indices_1]
    return subsampled_matrix.data
def _subsample_embeddings_dim0(matrix_original, matrix_target, reduction_factor):
    indices = torch.arange(0, matrix_original.size(0), reduction_factor)
    out_dim_0 = int(indices.shape[0])
    target_d0 = int(matrix_target.shape[0])
    if out_dim_0 == target_d0:
        pass
    else:
        indices = indices[:target_d0]
    subsampled_matrix = matrix_original[indices, :]
    return subsampled_matrix.data
def _subsample_embeddings_dim1(matrix_original, matrix_target, reduction_factor):
    indices = torch.arange(0, matrix_original.size(1), reduction_factor)
    out_dim_1 = int(indices.shape[0])
    target_d1 = int(matrix_target.shape[1])
    if out_dim_1 == target_d1:
        pass
    else:
        indices = indices[:target_d1]
    subsampled_matrix = matrix_original[:, indices]
    return subsampled_matrix.data
def _subsample_embeddings_dimlast(matrix_original, matrix_target, reduction_factor):
    device = matrix_original.get_device() 
    indices = torch.arange(0, matrix_original.size(-1), reduction_factor).to(device)
    out_dim_1 = int(indices.shape[0])
    target_d1 = int(matrix_target.shape[-1])
    if out_dim_1 == target_d1:
       pass
    else:
        indices = indices[:target_d1]
    #subsampled_matrix = matrix_original[:, :, indices]
    subsampled_matrix = matrix_original.index_select(dim=-1, index=indices)
    return subsampled_matrix.data


# make the smaller model 
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
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.llm,
        #     revision = self.revision,
        #     cache_dir = self.cache_dir,
        #     trust_remote_code=True
        # )


        self.model_config = AutoConfig.from_pretrained(self.llm, trust_remote_code=True)
        # self.model_config.rotary_dim = int(self.model_config.rotary_dim / self.reduction_factor)
        self.model_config.n_embd = int(self.model_config.n_embd / self.reduction_factor)
        self.model_config.n_inner = int(self.model_config.n_inner / self.reduction_factor)

        self.model = AutoModelForCausalLM.from_config(
            self.model_config,
            trust_remote_code=True
        )

        model_copy = copy.deepcopy(self.model)


        
        # downsampling the weights
        self.reduce() # ask --> cannot pre-define the framework config

        # print("======")
        # print(model_copy.transformer.wte.weight.data[0][:10])
        # print(self.model.transformer.wte.weight.data[0][:10])
        # print("======")

        # for (name_original, param_original), (name, param) in zip(model_copy.named_parameters(), self.model.named_parameters()): 
        #     print(name_original, torch.equal(param_original, param)) 
        # exit()
        

    def reduce(self):
        # # Create a copy of the state_dict for modifications

        model_original = AutoModelForCausalLM.from_pretrained(
            self.llm,
            revision = self.revision,
            cache_dir = self.cache_dir,
            trust_remote_code=True
        )
        # If there are 10 param, could I use index to assign orinnt out the specific param instead of use for loop?
        state_dict = self.model.state_dict()
        for (name_original, param_original), (name, param) in zip(model_original.named_parameters(), self.model.named_parameters()):
            if param.dim() == 2:
                # 2D weight matrices
                if param.size(0) == param.size(1):
                    # Subsample and scale square matrices
                    #param.data = self._subsample_and_scale(param_original, param)
                    #state_dict[name] = _subsample_and_scale(param_original, param, self.reduction_factor)
                    param.data = _subsample_and_scale(param_original, param, self.reduction_factor)
                else:
                    # Handle rectangular matrices by subsampling only the larger dimension
                    if "wte" in name:
                        #param.data = self._subsample_embeddings_dim1(param_original, param)
                        #self.model.state_dict[name] = _subsample_embeddings_dim1(param_original, param, self.reduction_factor)
                        param.data = _subsample_embeddings_dim1(param_original, param, self.reduction_factor)
                    else:
                        #param.data = self._subsample_embeddings_dim(param_original, param)
                        #self.model.state_dict[name] = _subsample_embeddings_dim(param_original, param, self.reduction_factor)
                        param.data = _subsample_embeddings_dim(param_original, param, self.reduction_factor)
            else:
                # embedding, bias, .... (1D)
                #param.data = self._subsample_embeddings(param_original, param)
                #state_dict[name] = _subsample_embeddings(param_original, param, self.reduction_factor)
                param.data = _subsample_embeddings(param_original, param, self.reduction_factor)
        
        del model_original

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
        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.cpus,
        )
        self.eval_max_batch = math.ceil(self.loader.dataset.__eval_size__()/self.batch_size)
        self.train_max_batch = math.ceil(self.loader.dataset.__train_size__()/self.batch_size)

        #Save hook inf.
        self.smaller_hook_forward_dict = {}
        self.smaller_hook_backward_dict = {}

        self.larger_hook_forward_dict = {}
        self.larger_hook_backward_dict = {}

        #config = AutoConfig.from_pretrained('LLM360/CrystalCoder', trust_remote_code=True)
        #config.save_pretrained('../distill-crystalcoder-config')


        self.show_params(self.larger_model.model)
        self.show_params(self.smaller_model.model)


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

        #fsdp_plugin = FSDPPlugin(mixed_precision="bf16", reshard_after_forward=True)

        # I can change to fsdp
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.grad,
            #mixed_precision = 'fp8',
            mixed_precision = 'bf16',
            #fsdp_plugin = fsdp_plugin,
            #megatron_lm_plugin = True,
            #deepspeed_plugin = True,
        )

        self.larger_model, self.smaller_model, self.opt, self.loader= self.accelerator.prepare(
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
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        '''

    def compute_mean_loss(self, large_hidden_states, small_hidden_states):
        # print(large_hidden_states.shape) #torch.Size([layers of the model, batch_size, seq_length, output_dim])
        # print(small_hidden_states.shape) #torch.Size([layers of the model, batch_size, seq_length, output_dim])
        mean_large = large_hidden_states.mean(dim=(1, 2, 3))
        mean_small = small_hidden_states.mean(dim=(1, 2, 3))
        mean_loss = torch.abs(mean_large - mean_small).sum()
        return mean_loss

    def compute_kl_divergence_distance(self, large_hidden_states, small_hidden_states):
        # Small constant to prevent division by zero in std computation
        epsilon = 1e-5
        mean_large = large_hidden_states.mean(dim=(1, 2, 3), keepdim=True)
        std_large = large_hidden_states.std(dim=(1, 2, 3), keepdim=True) + epsilon
        mean_small = small_hidden_states.mean(dim=(1, 2, 3), keepdim=True)
        std_small = small_hidden_states.std(dim=(1, 2, 3), keepdim=True) + epsilon
         # Check for NaNs in computed means and stds
        if torch.isnan(mean_large).any() or torch.isnan(mean_small).any() or torch.isnan(std_large).any() or torch.isnan(std_small).any():
            print()
            print("KL_loss: NaN detected in statistical computations")
            # Return zero (or epsilon) loss or handle appropriately
            return torch.tensor(0.0, device=mean_large.device)  # Return zero loss or handle appropriately
        # build distribute
        dist_large = Normal(mean_large, std_large)
        dist_small = Normal(mean_small, std_small)
        # caculate KL by layer
        #kl_div = kl_divergence(dist_large, dist_small).mean(dim=1)
        kl_div = kl_divergence(dist_large, dist_small).mean(dim=(1, 2, 3)).sum()
        return kl_div

    def loss_layer(self, output_large, output_small):

        '''
        # Logit loss
        large_logits = output_large.logits #torch.Size([4, 2048, 32032])
        small_logits = output_small.logits #torch.Size([4, 2048, 32032])
        loss = torch.nn.MSELoss()(large_logits, small_logits)
        '''

        large_hidden_states = output_large.hidden_states
        large_hidden_states = torch.stack(large_hidden_states)
        small_hidden_states = output_small.hidden_states
        small_hidden_states = torch.stack(small_hidden_states)
        # print("======")
        # print(large_hidden_states.shape) #torch.Size([33, 4, 2048, 4096])
        # print(small_hidden_states.shape) #torch.Size([33, 4, 2048, 1024])
        # print(large_hidden_states.shape) #torch.Size([layers of the model, batch_size, seq_length, output_dim])
        # print(small_hidden_states.shape) #torch.Size([layers of the model, batch_size, seq_length, output_dim])
        # print("======")


        # compute_mean_loss of large_hidden_states and small_hidden_states for each layer
        # calculate the overall loss as the sum of mean_loss and wasserstein_distance
        mean_loss = self.compute_mean_loss(large_hidden_states, small_hidden_states)
        # compute_kl_distance of large_hidden_states and small_hidden_states for each layer
        kl_loss = self.compute_kl_divergence_distance(large_hidden_states, small_hidden_states)
        #print("ml:", mean_loss.shape)
        #print("kl:", kl_loss.shape)
        loss = mean_loss + kl_loss

        return loss


    def forward_hook(self, module_name, model_name):
        if model_name == "smaller":
            def f_hook(module, input, output):
                self.smaller_hook_forward_dict[module_name] = output
            return f_hook
        elif model_name == "larger":
            def f_hook(module, input, output):
                self.larger_hook_forward_dict[module_name] = output
            return f_hook
        else:
            raise ValueError(f"Error file: distill_llm.py, Invalid number: line 491+-")

    # normal --> clear hook and then do the llm pre-training loss
    # use the below caculate loss first. and then remove the hook to 
    def backward_hook(self, module_name, model_name):
        if model_name == "smaller":
            def b_hook(module, grad_input, grad_output, is_before=True):
                if is_before:
                    ######################
                    # new_grad_input_tensor = torch.randn(1, 10) 
                    # modified_grad_input = tuple([new_grad_input_tensor if g is not None else None for g in grad_input])
                    # print("New grad_input before backward:", modified_grad_input)
                    # return modified_grad_input
                    ######################
                    downsampled_grad_input = _subsample_embeddings_dimlast(self.larger_hook_backward_dict[module_name], grad_input, self.smaller_model.reduction_factor) 
                    return downsampled_grad_input
                    ######################
                    # return grad_input
                else:
                    self.smaller_hook_backward_dict[module_name] = grad_output
            return b_hook
        elif model_name == "larger":
            def b_hook(module, grad_input, grad_output, is_before=True):
                self.larger_hook_backward_dict[module_name] = grad_output
            return b_hook
        else:
            raise ValueError(f"Error file: distill_llm.py, Invalid number: line 504+-")
    
    
    # def backward_hook(self, module_name, model_name):
    #     if model_name == "smaller":
    #         def b_hook(module, grad_input, grad_output):
    #             self.smaller_hook_backward_dict[module_name] = grad_output
    #         return b_hook
    #     elif model_name == "larger":
    #         def b_hook(module, grad_input, grad_output):
    #             self.larger_hook_backward_dict[module_name] = grad_output
    #         return b_hook
    #     else:
    #         raise ValueError(f"Error file: distill_llm.py, Invalid number: line 504+-")


    def smaller_register_hook(self, model, model_name="smaller"):
        #for module_name, module in model.named_parameters():
        for module_name, module in model.named_modules():
            # weight, basi, embedding wte, needs?
            #if hasattr(module, 'weight'):
            module.register_forward_hook(self.forward_hook(module_name, model_name))
            module.register_full_backward_hook(self.backward_hook(module_name, model_name))

    def larger_register_hook(self, model, model_name="larger"):
        #for module_name, module in model.named_parameters():
        for module_name, module in model.named_modules():
            # weight, basi, embedding wte, needs?
            #if hasattr(module, 'weight'):
            module.register_forward_hook(self.forward_hook(module_name, model_name))
            module.register_full_backward_hook(self.backward_hook(module_name, model_name))

    def loss_hidden(self, output_large, output_small):
        ############
        large_hidden_states = output_large.hidden_states
        large_hidden_states = torch.stack(large_hidden_states)
        small_hidden_states = output_small.hidden_states
        small_hidden_states = torch.stack(small_hidden_states)


        # print(11111)
        # for name, module in model.named_modules():
        #     print("========")
        #     print(name, module)
        #     if hasattr(module, 'weight'):
        #         module.register_forward_hook(self.hook_function)
        #     else:
        #         print("fail:")
        #         print(name)
        #         print("----")
        #         print(module)
        # print(11111)

        # import pdb; pdb.set_trace()
        # exit()
        
        
        
        #for name, module in model.named_parameters():
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight'):
        #         return module.register_forward_hook(self.hook_function)
        #     else:
        #         print("fail:", name)
    
    def forward_loss(self, larger_model, smaller_model, larger_hook_forward_dict, smaller_hook_forward_dict):
        #block-level now --> module-level --> matric weight level 

        # !!!!!! because dict is on cpu, so I need to move it to gpu !!!!!!!!!!!!!! Need to change to tensor
        # Create downsample x: Downsample larger_model hidden_state * W^T_(smaller_model) to get x 
        loss = 0
        for layer in range(len(self.smaller_model.model.transformer.h)):
            if layer == 0:
                downsampled_x = _subsample_embeddings_dimlast(larger_hook_forward_dict['transformer.wte'], smaller_hook_forward_dict['transformer.wte'], self.smaller_model.reduction_factor)
                #downsampled_x = smaller_model.model.drop(downsampled_x)
                #downsampled_x = _subsample_embeddings_dimlast(larger_hook_forward_dict['transformer.drop'], smaller_hook_forward_dict['transformer.drop'], self.smaller_model.reduction_factor)
                y_prime = smaller_model.model.transformer.h[layer](downsampled_x)[0]
                y = smaller_hook_forward_dict[f"transformer.h.{layer}"][0]
                f"transformer.h.{layer-1}"
            else:  
                downsampled_x = _subsample_embeddings_dimlast(larger_hook_forward_dict[f"transformer.h.{layer-1}"][0], smaller_hook_forward_dict[f"transformer.h.{layer-1}"][0], self.smaller_model.reduction_factor)
                y_prime = smaller_model.model.transformer.h[layer](downsampled_x)[0]
                y = smaller_hook_forward_dict['transformer.h'+"."+str(layer)][0]

            #loss += F.kl_div(y_prime.log(), y, reduction='batchmean')
            loss += torch.nn.MSELoss()(y_prime, y)
        return loss
        
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

        self.larger_model.model.to(self.device)
        self.smaller_model.model.to(self.device)


        # Need to revise
        #half: fp16
        ######
        self.larger_model.model.half()
        self.smaller_model.model.half()

        self.larger_model.model.train()
        self.smaller_model.model.train()
        self.smaller_model.model.train()
        ######

        print()
        print("lr:{}".format(self.learning_rate))
        loss_1 = 0
        loss_2 = 0

        self.smaller_register_hook(self.smaller_model.model)
        self.larger_register_hook(self.larger_model.model)
        
        #loss_1 --> forward pass: weight matrix
        #loss_2 --> backward pass: weight matrix
        #loss_3 --> hidden state difference
        for i, batch in enumerate(prog):
            
            #
            self.opt.zero_grad()

            if i == stop_batch:
                break
            self.step = i + 1

            output_large = self.larger_model.forward(batch)
            #output_small = self.smaller_model.forward(batch)

            # for name, param in self.smaller_hook_backward_dict.items():
            #     print(name, len(param))
            
            # import pdb; pdb.set_trace()
            
            # exit()
            
            # step1: backward_loss
            
            
            
            # step1: forward_loss
            output_small = self.smaller_model.forward(batch)
            forward_loss = self.forward_loss(self.larger_model, self.smaller_model, self.larger_hook_forward_dict, self.smaller_hook_forward_dict)
            
            # step2:

            # step3:
            #loss_1 = self.loss_hidden(output_large, output_small)

            # total:
            #loss = loss_1
            loss = forward_loss

            stop_batch=10
            print(loss)
            
            #accumulated_loss += loss_2.item()
            #total_loss += loss_2.item()
            prog.set_description(f"loss: {loss.item():.3f}")
            '''
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=i,
            )
            '''

            self.accelerator.backward(loss)

            '''
            ####################
            # check whether optimize the parmaeters?
            for initial_param, trained_param in zip(self.larger_model.model.parameters(), self.smaller_model.model.parameters()):
                if not torch.equal(initial_param, trained_param):
                    print("Parameter changed!")
                else:
                    print("Parameter unchanged.")
            ####################
            '''

            #self.accelerator.clip_grad_norm_(self.smaller_model.model.parameters(), max_norm=1.0)

            self.opt.step()

            # clean self.larger_model gredient
            with torch.no_grad():
                self.larger_model.model.zero_grad()
            
            # Clean hook:
            self.larger_hook_forward_dict.clear()
            self.larger_hook_backward_dict.clear()
            self.smaller_hook_forward_dict.clear()
            self.smaller_hook_backward_dict.clear()
            
            # Register hook:
            self.smaller_register_hook(self.smaller_model.model)
            self.larger_register_hook(self.larger_model.model)

            continue

            if (i + 1) % self.grad == 0:
                self.opt.step()
                prog.set_description(f"accumulated_loss: {total_loss:.3f}")
                accumulated_loss = 0.0

        total_loss = total_loss/self.step
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








# y-bias

'''
transformer.wte.weight torch.Size([32032, 1024])
transformer.h.0.ln_1.weight torch.Size([1024])
transformer.h.0.ln_1.bias torch.Size([1024])
transformer.h.0.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.0.attn.c_attn.bias torch.Size([3072])
transformer.h.0.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.0.attn.c_proj.bias torch.Size([1024])
transformer.h.0.ln_2.weight torch.Size([1024])
transformer.h.0.ln_2.bias torch.Size([1024])
transformer.h.0.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.0.mlp.c_fc.bias torch.Size([2730])
transformer.h.0.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.0.mlp.c_fc2.bias torch.Size([2730])
transformer.h.0.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.0.mlp.c_proj.bias torch.Size([1024])
transformer.h.1.ln_1.weight torch.Size([1024])
transformer.h.1.ln_1.bias torch.Size([1024])
transformer.h.1.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.1.attn.c_attn.bias torch.Size([3072])
transformer.h.1.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.1.attn.c_proj.bias torch.Size([1024])
transformer.h.1.ln_2.weight torch.Size([1024])
transformer.h.1.ln_2.bias torch.Size([1024])
transformer.h.1.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.1.mlp.c_fc.bias torch.Size([2730])
transformer.h.1.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.1.mlp.c_fc2.bias torch.Size([2730])
transformer.h.1.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.1.mlp.c_proj.bias torch.Size([1024])
transformer.h.2.ln_1.weight torch.Size([1024])
transformer.h.2.ln_1.bias torch.Size([1024])
transformer.h.2.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.2.attn.c_attn.bias torch.Size([3072])
transformer.h.2.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.2.attn.c_proj.bias torch.Size([1024])
transformer.h.2.ln_2.weight torch.Size([1024])
transformer.h.2.ln_2.bias torch.Size([1024])
transformer.h.2.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.2.mlp.c_fc.bias torch.Size([2730])
transformer.h.2.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.2.mlp.c_fc2.bias torch.Size([2730])
transformer.h.2.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.2.mlp.c_proj.bias torch.Size([1024])
transformer.h.3.ln_1.weight torch.Size([1024])
transformer.h.3.ln_1.bias torch.Size([1024])
transformer.h.3.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.3.attn.c_attn.bias torch.Size([3072])
transformer.h.3.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.3.attn.c_proj.bias torch.Size([1024])
transformer.h.3.ln_2.weight torch.Size([1024])
transformer.h.3.ln_2.bias torch.Size([1024])
transformer.h.3.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.3.mlp.c_fc.bias torch.Size([2730])
transformer.h.3.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.3.mlp.c_fc2.bias torch.Size([2730])
transformer.h.3.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.3.mlp.c_proj.bias torch.Size([1024])
transformer.h.4.ln_1.weight torch.Size([1024])
transformer.h.4.ln_1.bias torch.Size([1024])
transformer.h.4.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.4.attn.c_attn.bias torch.Size([3072])
transformer.h.4.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.4.attn.c_proj.bias torch.Size([1024])
transformer.h.4.ln_2.weight torch.Size([1024])
transformer.h.4.ln_2.bias torch.Size([1024])
transformer.h.4.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.4.mlp.c_fc.bias torch.Size([2730])
transformer.h.4.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.4.mlp.c_fc2.bias torch.Size([2730])
transformer.h.4.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.4.mlp.c_proj.bias torch.Size([1024])
transformer.h.5.ln_1.weight torch.Size([1024])
transformer.h.5.ln_1.bias torch.Size([1024])
transformer.h.5.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.5.attn.c_attn.bias torch.Size([3072])
transformer.h.5.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.5.attn.c_proj.bias torch.Size([1024])
transformer.h.5.ln_2.weight torch.Size([1024])
transformer.h.5.ln_2.bias torch.Size([1024])
transformer.h.5.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.5.mlp.c_fc.bias torch.Size([2730])
transformer.h.5.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.5.mlp.c_fc2.bias torch.Size([2730])
transformer.h.5.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.5.mlp.c_proj.bias torch.Size([1024])
transformer.h.6.ln_1.weight torch.Size([1024])
transformer.h.6.ln_1.bias torch.Size([1024])
transformer.h.6.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.6.attn.c_attn.bias torch.Size([3072])
transformer.h.6.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.6.attn.c_proj.bias torch.Size([1024])
transformer.h.6.ln_2.weight torch.Size([1024])
transformer.h.6.ln_2.bias torch.Size([1024])
transformer.h.6.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.6.mlp.c_fc.bias torch.Size([2730])
transformer.h.6.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.6.mlp.c_fc2.bias torch.Size([2730])
transformer.h.6.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.6.mlp.c_proj.bias torch.Size([1024])
transformer.h.7.ln_1.weight torch.Size([1024])
transformer.h.7.ln_1.bias torch.Size([1024])
transformer.h.7.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.7.attn.c_attn.bias torch.Size([3072])
transformer.h.7.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.7.attn.c_proj.bias torch.Size([1024])
transformer.h.7.ln_2.weight torch.Size([1024])
transformer.h.7.ln_2.bias torch.Size([1024])
transformer.h.7.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.7.mlp.c_fc.bias torch.Size([2730])
transformer.h.7.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.7.mlp.c_fc2.bias torch.Size([2730])
transformer.h.7.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.7.mlp.c_proj.bias torch.Size([1024])
transformer.h.8.ln_1.weight torch.Size([1024])
transformer.h.8.ln_1.bias torch.Size([1024])
transformer.h.8.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.8.attn.c_attn.bias torch.Size([3072])
transformer.h.8.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.8.attn.c_proj.bias torch.Size([1024])
transformer.h.8.ln_2.weight torch.Size([1024])
transformer.h.8.ln_2.bias torch.Size([1024])
transformer.h.8.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.8.mlp.c_fc.bias torch.Size([2730])
transformer.h.8.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.8.mlp.c_fc2.bias torch.Size([2730])
transformer.h.8.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.8.mlp.c_proj.bias torch.Size([1024])
transformer.h.9.ln_1.weight torch.Size([1024])
transformer.h.9.ln_1.bias torch.Size([1024])
transformer.h.9.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.9.attn.c_attn.bias torch.Size([3072])
transformer.h.9.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.9.attn.c_proj.bias torch.Size([1024])
transformer.h.9.ln_2.weight torch.Size([1024])
transformer.h.9.ln_2.bias torch.Size([1024])
transformer.h.9.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.9.mlp.c_fc.bias torch.Size([2730])
transformer.h.9.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.9.mlp.c_fc2.bias torch.Size([2730])
transformer.h.9.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.9.mlp.c_proj.bias torch.Size([1024])
transformer.h.10.ln_1.weight torch.Size([1024])
transformer.h.10.ln_1.bias torch.Size([1024])
transformer.h.10.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.10.attn.c_attn.bias torch.Size([3072])
transformer.h.10.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.10.attn.c_proj.bias torch.Size([1024])
transformer.h.10.ln_2.weight torch.Size([1024])
transformer.h.10.ln_2.bias torch.Size([1024])
transformer.h.10.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.10.mlp.c_fc.bias torch.Size([2730])
transformer.h.10.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.10.mlp.c_fc2.bias torch.Size([2730])
transformer.h.10.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.10.mlp.c_proj.bias torch.Size([1024])
transformer.h.11.ln_1.weight torch.Size([1024])
transformer.h.11.ln_1.bias torch.Size([1024])
transformer.h.11.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.11.attn.c_attn.bias torch.Size([3072])
transformer.h.11.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.11.attn.c_proj.bias torch.Size([1024])
transformer.h.11.ln_2.weight torch.Size([1024])
transformer.h.11.ln_2.bias torch.Size([1024])
transformer.h.11.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.11.mlp.c_fc.bias torch.Size([2730])
transformer.h.11.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.11.mlp.c_fc2.bias torch.Size([2730])
transformer.h.11.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.11.mlp.c_proj.bias torch.Size([1024])
transformer.h.12.ln_1.weight torch.Size([1024])
transformer.h.12.ln_1.bias torch.Size([1024])
transformer.h.12.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.12.attn.c_attn.bias torch.Size([3072])
transformer.h.12.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.12.attn.c_proj.bias torch.Size([1024])
transformer.h.12.ln_2.weight torch.Size([1024])
transformer.h.12.ln_2.bias torch.Size([1024])
transformer.h.12.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.12.mlp.c_fc.bias torch.Size([2730])
transformer.h.12.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.12.mlp.c_fc2.bias torch.Size([2730])
transformer.h.12.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.12.mlp.c_proj.bias torch.Size([1024])
transformer.h.13.ln_1.weight torch.Size([1024])
transformer.h.13.ln_1.bias torch.Size([1024])
transformer.h.13.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.13.attn.c_attn.bias torch.Size([3072])
transformer.h.13.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.13.attn.c_proj.bias torch.Size([1024])
transformer.h.13.ln_2.weight torch.Size([1024])
transformer.h.13.ln_2.bias torch.Size([1024])
transformer.h.13.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.13.mlp.c_fc.bias torch.Size([2730])
transformer.h.13.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.13.mlp.c_fc2.bias torch.Size([2730])
transformer.h.13.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.13.mlp.c_proj.bias torch.Size([1024])
transformer.h.14.ln_1.weight torch.Size([1024])
transformer.h.14.ln_1.bias torch.Size([1024])
transformer.h.14.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.14.attn.c_attn.bias torch.Size([3072])
transformer.h.14.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.14.attn.c_proj.bias torch.Size([1024])
transformer.h.14.ln_2.weight torch.Size([1024])
transformer.h.14.ln_2.bias torch.Size([1024])
transformer.h.14.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.14.mlp.c_fc.bias torch.Size([2730])
transformer.h.14.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.14.mlp.c_fc2.bias torch.Size([2730])
transformer.h.14.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.14.mlp.c_proj.bias torch.Size([1024])
transformer.h.15.ln_1.weight torch.Size([1024])
transformer.h.15.ln_1.bias torch.Size([1024])
transformer.h.15.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.15.attn.c_attn.bias torch.Size([3072])
transformer.h.15.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.15.attn.c_proj.bias torch.Size([1024])
transformer.h.15.ln_2.weight torch.Size([1024])
transformer.h.15.ln_2.bias torch.Size([1024])
transformer.h.15.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.15.mlp.c_fc.bias torch.Size([2730])
transformer.h.15.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.15.mlp.c_fc2.bias torch.Size([2730])
transformer.h.15.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.15.mlp.c_proj.bias torch.Size([1024])
transformer.h.16.ln_1.weight torch.Size([1024])
transformer.h.16.ln_1.bias torch.Size([1024])
transformer.h.16.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.16.attn.c_attn.bias torch.Size([3072])
transformer.h.16.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.16.attn.c_proj.bias torch.Size([1024])
transformer.h.16.ln_2.weight torch.Size([1024])
transformer.h.16.ln_2.bias torch.Size([1024])
transformer.h.16.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.16.mlp.c_fc.bias torch.Size([2730])
transformer.h.16.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.16.mlp.c_fc2.bias torch.Size([2730])
transformer.h.16.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.16.mlp.c_proj.bias torch.Size([1024])
transformer.h.17.ln_1.weight torch.Size([1024])
transformer.h.17.ln_1.bias torch.Size([1024])
transformer.h.17.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.17.attn.c_attn.bias torch.Size([3072])
transformer.h.17.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.17.attn.c_proj.bias torch.Size([1024])
transformer.h.17.ln_2.weight torch.Size([1024])
transformer.h.17.ln_2.bias torch.Size([1024])
transformer.h.17.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.17.mlp.c_fc.bias torch.Size([2730])
transformer.h.17.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.17.mlp.c_fc2.bias torch.Size([2730])
transformer.h.17.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.17.mlp.c_proj.bias torch.Size([1024])
transformer.h.18.ln_1.weight torch.Size([1024])
transformer.h.18.ln_1.bias torch.Size([1024])
transformer.h.18.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.18.attn.c_attn.bias torch.Size([3072])
transformer.h.18.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.18.attn.c_proj.bias torch.Size([1024])
transformer.h.18.ln_2.weight torch.Size([1024])
transformer.h.18.ln_2.bias torch.Size([1024])
transformer.h.18.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.18.mlp.c_fc.bias torch.Size([2730])
transformer.h.18.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.18.mlp.c_fc2.bias torch.Size([2730])
transformer.h.18.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.18.mlp.c_proj.bias torch.Size([1024])
transformer.h.19.ln_1.weight torch.Size([1024])
transformer.h.19.ln_1.bias torch.Size([1024])
transformer.h.19.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.19.attn.c_attn.bias torch.Size([3072])
transformer.h.19.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.19.attn.c_proj.bias torch.Size([1024])
transformer.h.19.ln_2.weight torch.Size([1024])
transformer.h.19.ln_2.bias torch.Size([1024])
transformer.h.19.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.19.mlp.c_fc.bias torch.Size([2730])
transformer.h.19.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.19.mlp.c_fc2.bias torch.Size([2730])
transformer.h.19.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.19.mlp.c_proj.bias torch.Size([1024])
transformer.h.20.ln_1.weight torch.Size([1024])
transformer.h.20.ln_1.bias torch.Size([1024])
transformer.h.20.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.20.attn.c_attn.bias torch.Size([3072])
transformer.h.20.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.20.attn.c_proj.bias torch.Size([1024])
transformer.h.20.ln_2.weight torch.Size([1024])
transformer.h.20.ln_2.bias torch.Size([1024])
transformer.h.20.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.20.mlp.c_fc.bias torch.Size([2730])
transformer.h.20.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.20.mlp.c_fc2.bias torch.Size([2730])
transformer.h.20.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.20.mlp.c_proj.bias torch.Size([1024])
transformer.h.21.ln_1.weight torch.Size([1024])
transformer.h.21.ln_1.bias torch.Size([1024])
transformer.h.21.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.21.attn.c_attn.bias torch.Size([3072])
transformer.h.21.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.21.attn.c_proj.bias torch.Size([1024])
transformer.h.21.ln_2.weight torch.Size([1024])
transformer.h.21.ln_2.bias torch.Size([1024])
transformer.h.21.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.21.mlp.c_fc.bias torch.Size([2730])
transformer.h.21.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.21.mlp.c_fc2.bias torch.Size([2730])
transformer.h.21.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.21.mlp.c_proj.bias torch.Size([1024])
transformer.h.22.ln_1.weight torch.Size([1024])
transformer.h.22.ln_1.bias torch.Size([1024])
transformer.h.22.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.22.attn.c_attn.bias torch.Size([3072])
transformer.h.22.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.22.attn.c_proj.bias torch.Size([1024])
transformer.h.22.ln_2.weight torch.Size([1024])
transformer.h.22.ln_2.bias torch.Size([1024])
transformer.h.22.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.22.mlp.c_fc.bias torch.Size([2730])
transformer.h.22.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.22.mlp.c_fc2.bias torch.Size([2730])
transformer.h.22.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.22.mlp.c_proj.bias torch.Size([1024])
transformer.h.23.ln_1.weight torch.Size([1024])
transformer.h.23.ln_1.bias torch.Size([1024])
transformer.h.23.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.23.attn.c_attn.bias torch.Size([3072])
transformer.h.23.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.23.attn.c_proj.bias torch.Size([1024])
transformer.h.23.ln_2.weight torch.Size([1024])
transformer.h.23.ln_2.bias torch.Size([1024])
transformer.h.23.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.23.mlp.c_fc.bias torch.Size([2730])
transformer.h.23.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.23.mlp.c_fc2.bias torch.Size([2730])
transformer.h.23.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.23.mlp.c_proj.bias torch.Size([1024])
transformer.h.24.ln_1.weight torch.Size([1024])
transformer.h.24.ln_1.bias torch.Size([1024])
transformer.h.24.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.24.attn.c_attn.bias torch.Size([3072])
transformer.h.24.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.24.attn.c_proj.bias torch.Size([1024])
transformer.h.24.ln_2.weight torch.Size([1024])
transformer.h.24.ln_2.bias torch.Size([1024])
transformer.h.24.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.24.mlp.c_fc.bias torch.Size([2730])
transformer.h.24.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.24.mlp.c_fc2.bias torch.Size([2730])
transformer.h.24.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.24.mlp.c_proj.bias torch.Size([1024])
transformer.h.25.ln_1.weight torch.Size([1024])
transformer.h.25.ln_1.bias torch.Size([1024])
transformer.h.25.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.25.attn.c_attn.bias torch.Size([3072])
transformer.h.25.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.25.attn.c_proj.bias torch.Size([1024])
transformer.h.25.ln_2.weight torch.Size([1024])
transformer.h.25.ln_2.bias torch.Size([1024])
transformer.h.25.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.25.mlp.c_fc.bias torch.Size([2730])
transformer.h.25.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.25.mlp.c_fc2.bias torch.Size([2730])
transformer.h.25.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.25.mlp.c_proj.bias torch.Size([1024])
transformer.h.26.ln_1.weight torch.Size([1024])
transformer.h.26.ln_1.bias torch.Size([1024])
transformer.h.26.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.26.attn.c_attn.bias torch.Size([3072])
transformer.h.26.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.26.attn.c_proj.bias torch.Size([1024])
transformer.h.26.ln_2.weight torch.Size([1024])
transformer.h.26.ln_2.bias torch.Size([1024])
transformer.h.26.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.26.mlp.c_fc.bias torch.Size([2730])
transformer.h.26.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.26.mlp.c_fc2.bias torch.Size([2730])
transformer.h.26.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.26.mlp.c_proj.bias torch.Size([1024])
transformer.h.27.ln_1.weight torch.Size([1024])
transformer.h.27.ln_1.bias torch.Size([1024])
transformer.h.27.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.27.attn.c_attn.bias torch.Size([3072])
transformer.h.27.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.27.attn.c_proj.bias torch.Size([1024])
transformer.h.27.ln_2.weight torch.Size([1024])
transformer.h.27.ln_2.bias torch.Size([1024])
transformer.h.27.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.27.mlp.c_fc.bias torch.Size([2730])
transformer.h.27.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.27.mlp.c_fc2.bias torch.Size([2730])
transformer.h.27.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.27.mlp.c_proj.bias torch.Size([1024])
transformer.h.28.ln_1.weight torch.Size([1024])
transformer.h.28.ln_1.bias torch.Size([1024])
transformer.h.28.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.28.attn.c_attn.bias torch.Size([3072])
transformer.h.28.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.28.attn.c_proj.bias torch.Size([1024])
transformer.h.28.ln_2.weight torch.Size([1024])
transformer.h.28.ln_2.bias torch.Size([1024])
transformer.h.28.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.28.mlp.c_fc.bias torch.Size([2730])
transformer.h.28.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.28.mlp.c_fc2.bias torch.Size([2730])
transformer.h.28.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.28.mlp.c_proj.bias torch.Size([1024])
transformer.h.29.ln_1.weight torch.Size([1024])
transformer.h.29.ln_1.bias torch.Size([1024])
transformer.h.29.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.29.attn.c_attn.bias torch.Size([3072])
transformer.h.29.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.29.attn.c_proj.bias torch.Size([1024])
transformer.h.29.ln_2.weight torch.Size([1024])
transformer.h.29.ln_2.bias torch.Size([1024])
transformer.h.29.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.29.mlp.c_fc.bias torch.Size([2730])
transformer.h.29.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.29.mlp.c_fc2.bias torch.Size([2730])
transformer.h.29.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.29.mlp.c_proj.bias torch.Size([1024])
transformer.h.30.ln_1.weight torch.Size([1024])
transformer.h.30.ln_1.bias torch.Size([1024])
transformer.h.30.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.30.attn.c_attn.bias torch.Size([3072])
transformer.h.30.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.30.attn.c_proj.bias torch.Size([1024])
transformer.h.30.ln_2.weight torch.Size([1024])
transformer.h.30.ln_2.bias torch.Size([1024])
transformer.h.30.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.30.mlp.c_fc.bias torch.Size([2730])
transformer.h.30.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.30.mlp.c_fc2.bias torch.Size([2730])
transformer.h.30.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.30.mlp.c_proj.bias torch.Size([1024])
transformer.h.31.ln_1.weight torch.Size([1024])
transformer.h.31.ln_1.bias torch.Size([1024])
transformer.h.31.attn.c_attn.weight torch.Size([1024, 3072])
transformer.h.31.attn.c_attn.bias torch.Size([3072])
transformer.h.31.attn.c_proj.weight torch.Size([1024, 1024])
transformer.h.31.attn.c_proj.bias torch.Size([1024])
transformer.h.31.ln_2.weight torch.Size([1024])
transformer.h.31.ln_2.bias torch.Size([1024])
transformer.h.31.mlp.c_fc.weight torch.Size([1024, 2730])
transformer.h.31.mlp.c_fc.bias torch.Size([2730])
transformer.h.31.mlp.c_fc2.weight torch.Size([1024, 2730])
transformer.h.31.mlp.c_fc2.bias torch.Size([2730])
transformer.h.31.mlp.c_proj.weight torch.Size([2730, 1024])
transformer.h.31.mlp.c_proj.bias torch.Size([1024])
transformer.ln_f.weight torch.Size([1024])
transformer.ln_f.bias torch.Size([1024])
'''

'''
transformer.wte 1
transformer.drop 1
transformer.h.0.ln_1 1
transformer.h.0.attn.c_attn 1
transformer.h.0.attn.attn_dropout 1
transformer.h.0.attn.c_proj 1
transformer.h.0.attn.resid_dropout 1
transformer.h.0.attn 2
transformer.h.0.ln_2 1
transformer.h.0.mlp.c_fc2 1
transformer.h.0.mlp.c_fc 1
transformer.h.0.mlp.act 1
transformer.h.0.mlp.c_proj 1
transformer.h.0.mlp.dropout 1
transformer.h.0.mlp 1
transformer.h.0 2
transformer.h.1.ln_1 1
transformer.h.1.attn.c_attn 1
transformer.h.1.attn.attn_dropout 1
transformer.h.1.attn.c_proj 1
transformer.h.1.attn.resid_dropout 1
transformer.h.1.attn 2
transformer.h.1.ln_2 1
transformer.h.1.mlp.c_fc2 1
transformer.h.1.mlp.c_fc 1
transformer.h.1.mlp.act 1
transformer.h.1.mlp.c_proj 1
transformer.h.1.mlp.dropout 1
transformer.h.1.mlp 1
transformer.h.1 2
transformer.h.2.ln_1 1
transformer.h.2.attn.c_attn 1
transformer.h.2.attn.attn_dropout 1
transformer.h.2.attn.c_proj 1
transformer.h.2.attn.resid_dropout 1
transformer.h.2.attn 2
transformer.h.2.ln_2 1
transformer.h.2.mlp.c_fc2 1
transformer.h.2.mlp.c_fc 1
transformer.h.2.mlp.act 1
transformer.h.2.mlp.c_proj 1
transformer.h.2.mlp.dropout 1
transformer.h.2.mlp 1
transformer.h.2 2
transformer.h.3.ln_1 1
transformer.h.3.attn.c_attn 1
transformer.h.3.attn.attn_dropout 1
transformer.h.3.attn.c_proj 1
transformer.h.3.attn.resid_dropout 1
transformer.h.3.attn 2
transformer.h.3.ln_2 1
transformer.h.3.mlp.c_fc2 1
transformer.h.3.mlp.c_fc 1
transformer.h.3.mlp.act 1
transformer.h.3.mlp.c_proj 1
transformer.h.3.mlp.dropout 1
transformer.h.3.mlp 1
transformer.h.3 2
transformer.h.4.ln_1 1
transformer.h.4.attn.c_attn 1
transformer.h.4.attn.attn_dropout 1
transformer.h.4.attn.c_proj 1
transformer.h.4.attn.resid_dropout 1
transformer.h.4.attn 2
transformer.h.4.ln_2 1
transformer.h.4.mlp.c_fc2 1
transformer.h.4.mlp.c_fc 1
transformer.h.4.mlp.act 1
transformer.h.4.mlp.c_proj 1
transformer.h.4.mlp.dropout 1
transformer.h.4.mlp 1
transformer.h.4 2
transformer.h.5.ln_1 1
transformer.h.5.attn.c_attn 1
transformer.h.5.attn.attn_dropout 1
transformer.h.5.attn.c_proj 1
transformer.h.5.attn.resid_dropout 1
transformer.h.5.attn 2
transformer.h.5.ln_2 1
transformer.h.5.mlp.c_fc2 1
transformer.h.5.mlp.c_fc 1
transformer.h.5.mlp.act 1
transformer.h.5.mlp.c_proj 1
transformer.h.5.mlp.dropout 1
transformer.h.5.mlp 1
transformer.h.5 2
transformer.h.6.ln_1 1
transformer.h.6.attn.c_attn 1
transformer.h.6.attn.attn_dropout 1
transformer.h.6.attn.c_proj 1
transformer.h.6.attn.resid_dropout 1
transformer.h.6.attn 2
transformer.h.6.ln_2 1
transformer.h.6.mlp.c_fc2 1
transformer.h.6.mlp.c_fc 1
transformer.h.6.mlp.act 1
transformer.h.6.mlp.c_proj 1
transformer.h.6.mlp.dropout 1
transformer.h.6.mlp 1
transformer.h.6 2
transformer.h.7.ln_1 1
transformer.h.7.attn.c_attn 1
transformer.h.7.attn.attn_dropout 1
transformer.h.7.attn.c_proj 1
transformer.h.7.attn.resid_dropout 1
transformer.h.7.attn 2
transformer.h.7.ln_2 1
transformer.h.7.mlp.c_fc2 1
transformer.h.7.mlp.c_fc 1
transformer.h.7.mlp.act 1
transformer.h.7.mlp.c_proj 1
transformer.h.7.mlp.dropout 1
transformer.h.7.mlp 1
transformer.h.7 2
transformer.h.8.ln_1 1
transformer.h.8.attn.c_attn 1
transformer.h.8.attn.attn_dropout 1
transformer.h.8.attn.c_proj 1
transformer.h.8.attn.resid_dropout 1
transformer.h.8.attn 2
transformer.h.8.ln_2 1
transformer.h.8.mlp.c_fc2 1
transformer.h.8.mlp.c_fc 1
transformer.h.8.mlp.act 1
transformer.h.8.mlp.c_proj 1
transformer.h.8.mlp.dropout 1
transformer.h.8.mlp 1
transformer.h.8 2
transformer.h.9.ln_1 1
transformer.h.9.attn.c_attn 1
transformer.h.9.attn.attn_dropout 1
transformer.h.9.attn.c_proj 1
transformer.h.9.attn.resid_dropout 1
transformer.h.9.attn 2
transformer.h.9.ln_2 1
transformer.h.9.mlp.c_fc2 1
transformer.h.9.mlp.c_fc 1
transformer.h.9.mlp.act 1
transformer.h.9.mlp.c_proj 1
transformer.h.9.mlp.dropout 1
transformer.h.9.mlp 1
transformer.h.9 2
transformer.h.10.ln_1 1
transformer.h.10.attn.c_attn 1
transformer.h.10.attn.attn_dropout 1
transformer.h.10.attn.c_proj 1
transformer.h.10.attn.resid_dropout 1
transformer.h.10.attn 2
transformer.h.10.ln_2 1
transformer.h.10.mlp.c_fc2 1
transformer.h.10.mlp.c_fc 1
transformer.h.10.mlp.act 1
transformer.h.10.mlp.c_proj 1
transformer.h.10.mlp.dropout 1
transformer.h.10.mlp 1
transformer.h.10 2
transformer.h.11.ln_1 1
transformer.h.11.attn.c_attn 1
transformer.h.11.attn.attn_dropout 1
transformer.h.11.attn.c_proj 1
transformer.h.11.attn.resid_dropout 1
transformer.h.11.attn 2
transformer.h.11.ln_2 1
transformer.h.11.mlp.c_fc2 1
transformer.h.11.mlp.c_fc 1
transformer.h.11.mlp.act 1
transformer.h.11.mlp.c_proj 1
transformer.h.11.mlp.dropout 1
transformer.h.11.mlp 1
transformer.h.11 2
transformer.h.12.ln_1 1
transformer.h.12.attn.c_attn 1
transformer.h.12.attn.attn_dropout 1
transformer.h.12.attn.c_proj 1
transformer.h.12.attn.resid_dropout 1
transformer.h.12.attn 2
transformer.h.12.ln_2 1
transformer.h.12.mlp.c_fc2 1
transformer.h.12.mlp.c_fc 1
transformer.h.12.mlp.act 1
transformer.h.12.mlp.c_proj 1
transformer.h.12.mlp.dropout 1
transformer.h.12.mlp 1
transformer.h.12 2
transformer.h.13.ln_1 1
transformer.h.13.attn.c_attn 1
transformer.h.13.attn.attn_dropout 1
transformer.h.13.attn.c_proj 1
transformer.h.13.attn.resid_dropout 1
transformer.h.13.attn 2
transformer.h.13.ln_2 1
transformer.h.13.mlp.c_fc2 1
transformer.h.13.mlp.c_fc 1
transformer.h.13.mlp.act 1
transformer.h.13.mlp.c_proj 1
transformer.h.13.mlp.dropout 1
transformer.h.13.mlp 1
transformer.h.13 2
transformer.h.14.ln_1 1
transformer.h.14.attn.c_attn 1
transformer.h.14.attn.attn_dropout 1
transformer.h.14.attn.c_proj 1
transformer.h.14.attn.resid_dropout 1
transformer.h.14.attn 2
transformer.h.14.ln_2 1
transformer.h.14.mlp.c_fc2 1
transformer.h.14.mlp.c_fc 1
transformer.h.14.mlp.act 1
transformer.h.14.mlp.c_proj 1
transformer.h.14.mlp.dropout 1
transformer.h.14.mlp 1
transformer.h.14 2
transformer.h.15.ln_1 1
transformer.h.15.attn.c_attn 1
transformer.h.15.attn.attn_dropout 1
transformer.h.15.attn.c_proj 1
transformer.h.15.attn.resid_dropout 1
transformer.h.15.attn 2
transformer.h.15.ln_2 1
transformer.h.15.mlp.c_fc2 1
transformer.h.15.mlp.c_fc 1
transformer.h.15.mlp.act 1
transformer.h.15.mlp.c_proj 1
transformer.h.15.mlp.dropout 1
transformer.h.15.mlp 1
transformer.h.15 2
transformer.h.16.ln_1 1
transformer.h.16.attn.c_attn 1
transformer.h.16.attn.attn_dropout 1
transformer.h.16.attn.c_proj 1
transformer.h.16.attn.resid_dropout 1
transformer.h.16.attn 2
transformer.h.16.ln_2 1
transformer.h.16.mlp.c_fc2 1
transformer.h.16.mlp.c_fc 1
transformer.h.16.mlp.act 1
transformer.h.16.mlp.c_proj 1
transformer.h.16.mlp.dropout 1
transformer.h.16.mlp 1
transformer.h.16 2
transformer.h.17.ln_1 1
transformer.h.17.attn.c_attn 1
transformer.h.17.attn.attn_dropout 1
transformer.h.17.attn.c_proj 1
transformer.h.17.attn.resid_dropout 1
transformer.h.17.attn 2
transformer.h.17.ln_2 1
transformer.h.17.mlp.c_fc2 1
transformer.h.17.mlp.c_fc 1
transformer.h.17.mlp.act 1
transformer.h.17.mlp.c_proj 1
transformer.h.17.mlp.dropout 1
transformer.h.17.mlp 1
transformer.h.17 2
transformer.h.18.ln_1 1
transformer.h.18.attn.c_attn 1
transformer.h.18.attn.attn_dropout 1
transformer.h.18.attn.c_proj 1
transformer.h.18.attn.resid_dropout 1
transformer.h.18.attn 2
transformer.h.18.ln_2 1
transformer.h.18.mlp.c_fc2 1
transformer.h.18.mlp.c_fc 1
transformer.h.18.mlp.act 1
transformer.h.18.mlp.c_proj 1
transformer.h.18.mlp.dropout 1
transformer.h.18.mlp 1
transformer.h.18 2
transformer.h.19.ln_1 1
transformer.h.19.attn.c_attn 1
transformer.h.19.attn.attn_dropout 1
transformer.h.19.attn.c_proj 1
transformer.h.19.attn.resid_dropout 1
transformer.h.19.attn 2
transformer.h.19.ln_2 1
transformer.h.19.mlp.c_fc2 1
transformer.h.19.mlp.c_fc 1
transformer.h.19.mlp.act 1
transformer.h.19.mlp.c_proj 1
transformer.h.19.mlp.dropout 1
transformer.h.19.mlp 1
transformer.h.19 2
transformer.h.20.ln_1 1
transformer.h.20.attn.c_attn 1
transformer.h.20.attn.attn_dropout 1
transformer.h.20.attn.c_proj 1
transformer.h.20.attn.resid_dropout 1
transformer.h.20.attn 2
transformer.h.20.ln_2 1
transformer.h.20.mlp.c_fc2 1
transformer.h.20.mlp.c_fc 1
transformer.h.20.mlp.act 1
transformer.h.20.mlp.c_proj 1
transformer.h.20.mlp.dropout 1
transformer.h.20.mlp 1
transformer.h.20 2
transformer.h.21.ln_1 1
transformer.h.21.attn.c_attn 1
transformer.h.21.attn.attn_dropout 1
transformer.h.21.attn.c_proj 1
transformer.h.21.attn.resid_dropout 1
transformer.h.21.attn 2
transformer.h.21.ln_2 1
transformer.h.21.mlp.c_fc2 1
transformer.h.21.mlp.c_fc 1
transformer.h.21.mlp.act 1
transformer.h.21.mlp.c_proj 1
transformer.h.21.mlp.dropout 1
transformer.h.21.mlp 1
transformer.h.21 2
transformer.h.22.ln_1 1
transformer.h.22.attn.c_attn 1
transformer.h.22.attn.attn_dropout 1
transformer.h.22.attn.c_proj 1
transformer.h.22.attn.resid_dropout 1
transformer.h.22.attn 2
transformer.h.22.ln_2 1
transformer.h.22.mlp.c_fc2 1
transformer.h.22.mlp.c_fc 1
transformer.h.22.mlp.act 1
transformer.h.22.mlp.c_proj 1
transformer.h.22.mlp.dropout 1
transformer.h.22.mlp 1
transformer.h.22 2
transformer.h.23.ln_1 1
transformer.h.23.attn.c_attn 1
transformer.h.23.attn.attn_dropout 1
transformer.h.23.attn.c_proj 1
transformer.h.23.attn.resid_dropout 1
transformer.h.23.attn 2
transformer.h.23.ln_2 1
transformer.h.23.mlp.c_fc2 1
transformer.h.23.mlp.c_fc 1
transformer.h.23.mlp.act 1
transformer.h.23.mlp.c_proj 1
transformer.h.23.mlp.dropout 1
transformer.h.23.mlp 1
transformer.h.23 2
transformer.h.24.ln_1 1
transformer.h.24.attn.c_attn 1
transformer.h.24.attn.attn_dropout 1
transformer.h.24.attn.c_proj 1
transformer.h.24.attn.resid_dropout 1
transformer.h.24.attn 2
transformer.h.24.ln_2 1
transformer.h.24.mlp.c_fc2 1
transformer.h.24.mlp.c_fc 1
transformer.h.24.mlp.act 1
transformer.h.24.mlp.c_proj 1
transformer.h.24.mlp.dropout 1
transformer.h.24.mlp 1
transformer.h.24 2
transformer.h.25.ln_1 1
transformer.h.25.attn.c_attn 1
transformer.h.25.attn.attn_dropout 1
transformer.h.25.attn.c_proj 1
transformer.h.25.attn.resid_dropout 1
transformer.h.25.attn 2
transformer.h.25.ln_2 1
transformer.h.25.mlp.c_fc2 1
transformer.h.25.mlp.c_fc 1
transformer.h.25.mlp.act 1
transformer.h.25.mlp.c_proj 1
transformer.h.25.mlp.dropout 1
transformer.h.25.mlp 1
transformer.h.25 2
transformer.h.26.ln_1 1
transformer.h.26.attn.c_attn 1
transformer.h.26.attn.attn_dropout 1
transformer.h.26.attn.c_proj 1
transformer.h.26.attn.resid_dropout 1
transformer.h.26.attn 2
transformer.h.26.ln_2 1
transformer.h.26.mlp.c_fc2 1
transformer.h.26.mlp.c_fc 1
transformer.h.26.mlp.act 1
transformer.h.26.mlp.c_proj 1
transformer.h.26.mlp.dropout 1
transformer.h.26.mlp 1
transformer.h.26 2
transformer.h.27.ln_1 1
transformer.h.27.attn.c_attn 1
transformer.h.27.attn.attn_dropout 1
transformer.h.27.attn.c_proj 1
transformer.h.27.attn.resid_dropout 1
transformer.h.27.attn 2
transformer.h.27.ln_2 1
transformer.h.27.mlp.c_fc2 1
transformer.h.27.mlp.c_fc 1
transformer.h.27.mlp.act 1
transformer.h.27.mlp.c_proj 1
transformer.h.27.mlp.dropout 1
transformer.h.27.mlp 1
transformer.h.27 2
transformer.h.28.ln_1 1
transformer.h.28.attn.c_attn 1
transformer.h.28.attn.attn_dropout 1
transformer.h.28.attn.c_proj 1
transformer.h.28.attn.resid_dropout 1
transformer.h.28.attn 2
transformer.h.28.ln_2 1
transformer.h.28.mlp.c_fc2 1
transformer.h.28.mlp.c_fc 1
transformer.h.28.mlp.act 1
transformer.h.28.mlp.c_proj 1
transformer.h.28.mlp.dropout 1
transformer.h.28.mlp 1
transformer.h.28 2
transformer.h.29.ln_1 1
transformer.h.29.attn.c_attn 1
transformer.h.29.attn.attn_dropout 1
transformer.h.29.attn.c_proj 1
transformer.h.29.attn.resid_dropout 1
transformer.h.29.attn 2
transformer.h.29.ln_2 1
transformer.h.29.mlp.c_fc2 1
transformer.h.29.mlp.c_fc 1
transformer.h.29.mlp.act 1
transformer.h.29.mlp.c_proj 1
transformer.h.29.mlp.dropout 1
transformer.h.29.mlp 1
transformer.h.29 2
transformer.h.30.ln_1 1
transformer.h.30.attn.c_attn 1
transformer.h.30.attn.attn_dropout 1
transformer.h.30.attn.c_proj 1
transformer.h.30.attn.resid_dropout 1
transformer.h.30.attn 2
transformer.h.30.ln_2 1
transformer.h.30.mlp.c_fc2 1
transformer.h.30.mlp.c_fc 1
transformer.h.30.mlp.act 1
transformer.h.30.mlp.c_proj 1
transformer.h.30.mlp.dropout 1
transformer.h.30.mlp 1
transformer.h.30 2
transformer.h.31.ln_1 1
transformer.h.31.attn.c_attn 1
transformer.h.31.attn.attn_dropout 1
transformer.h.31.attn.c_proj 1
transformer.h.31.attn.resid_dropout 1
transformer.h.31.attn 2
transformer.h.31.ln_2 1
transformer.h.31.mlp.c_fc2 1
transformer.h.31.mlp.c_fc 1
transformer.h.31.mlp.act 1
transformer.h.31.mlp.c_proj 1
transformer.h.31.mlp.dropout 1
transformer.h.31.mlp 1
transformer.h.31 2
transformer.ln_f 1
transformer 3
lm_head 1
 3
'''