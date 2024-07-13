"""exec(open('../pdb_test.py').read())"""

"""Write the code that you want to test here"""

print("\n")
print(f"==Test==")
print("\n")


grad_norms = {}
pattern = r"^transformer\.h\.\d{1,2}(?:\..+)?$"
for name, module in model.named_modules():
    layer_grad_norm = 0
    layer_name = ""
    # print(name)
    if name == "transformer.wte": 
        layer_name = name
    elif re.match(pattern, name):
        layer_name = '.'.join(name.split('.')[:3])
    else:
        continue
    for param in module.parameters():
        if param.grad is not None:
            #layer_grad_norm += torch.norm(param.grad.detach(), 2).item() ** 2
            if layer_name not in grad_norms:
                grad_norms[layer_name] = [torch.norm(param.grad.detach(), 2).item()]
            else:
                grad_norms[layer_name].append(torch.norm(param.grad.detach(), 2).item())
avg_grad_norms = {layer: sum(norms) / len(norms) for layer, norms in grad_norms.items()}



#print(grad_norms)
print(avg_grad_norms)


# def get_avg_grad_norm_per_layer(model):
#     grad_norms = {}
#     pattern_weight = r'\w+\.weight'
#     pattern_bias = r'\w+\.bias'
#     pattern = r"^transformer\.h\.\d{1,2}(?:\..+)?$"
#     #pattern = r"^transformer\.h\.\d{1,2}$"
#     for name, module in model.named_modules():
#         print(name)
#         layer_grad_norm = 0
#         layer_name = ""
#         if name == "transformer.wte": 
#             layer_name = name
#         elif re.match(pattern, name):
#             layer_name = '.'.join(name.split('.')[:3])
#         else:
#             continue
#             #print(name)
#             #raise ValueError("Wrong") 
#         for param in module.parameters():
#             if re.match(pattern_weight, name) or re.match(pattern_bias, name):
#                 if param.grad is not None:
#                     layer_grad_norm += torch.norm(param.grad.detach(), 2).item() ** 2
#             grad_norms[layer_name] = math.sqrt(layer_grad_norm)
#     #print(grad_norms)
#     return grad_norms


# def get_avg_grad_norm_per_layer(self, model):
#     # transformer.h.x 
#     # pattern = r"transformer\.h\.\d{1,2}$"
#     grad_norms = defaultdict(list)
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             #layer_name = name.split('.')[2]
#             layer_name = name
#             grad_norms[layer_name].append(torch.norm(param.grad.detach(), 2).item())
#     avg_grad_norms = {layer: sum(norms) / len(norms) for layer, norms in grad_norms.items()}
#     return avg_grad_norms


# def get_avg_grad_norm_per_layer(model):
#     # transformer.h.x 
#     # pattern = r"transformer\.h\.\d{1,2}$"
#     pattern = r"^transformer\.h\.\d{1,2}(?:\..+)?$"
#     grad_norms = defaultdict(list)
#     #pattern = r"transformer\.h\.\d{1,2}$"
#     layer_name = ""
#     for name, param in model.named_parameters():
#         if name == "transformer.wte.weight":
#             layer_name = name
#             layer_grad_norm += torch.norm(param.grad.detach(), 2).item() ** 2
#             grad_norms[layer_name] = math.sqrt(layer_grad_norm)
#         else:
#             if re.match(pattern, name):
#                 layer_name = '.'.join(name.split('.')[:2])
#                 layer_grad_norm = 0
#             elif "transformer.ln_f" in name:
#                 layer_name = "transformer.ln_f"
#                 layer_grad_norm = 0
#             else:
#                 print("Wrong")
                
#             if param.grad is not None:
#                 #layer_name = name.split('.')[2]
#                 #layer_name = name
#                 #grad_norms[layer_name].append(torch.norm(param.grad.detach(), 2).item())
#                 layer_grad_norm += torch.norm(param.grad.detach(), 2).item() ** 2
#     avg_grad_norms = {layer: sum(norms) / len(norms) for layer, norms in grad_norms.items()}
#     return avg_grad_norms



# def get_avg_grad_norm_per_layer(model):
#     for n, p in  



#get_avg_grad_norm_per_layer(self.smaller_model.model)
#avg_grad_norms = self.get_avg_grad_norm_per_layer(self.smaller_model.model)


#print(self.larger_hook_forward_dict.keys())


#print(len(self.larger_hook_forward_dict[module_name][0]))


#for k,v in self.larger_hook_forward_dict[module_name].items():



#input[0].data = _subsample_embeddings_dimlast(modified_input, input[0], self.smaller_model.reduction_factor)
#print(input)

# out = _subsample_embeddings_dimlast(modified_input, input_copy, self.smaller_model.reduction_factor)
# print(modified_input[0,0,:12])
# print(out[0,0,:12])
# print(input[0][0,0,:12])
# print()


# modified_input = self.larger_hook_forward_dict[module_name][0]
# print(modified_input, modified_input.shape, type(modified_input))

#input_copy = input[0].clone() 
#print(input_copy, input_copy.shape, type(input_copy))
# input[0].data = _subsample_embeddings_dimlast(modified_input, input, self.smaller_model.reduction_factor)  
# print(torch.equal(input_copy, input[0]))


# for k, v in self.larger_hook_forward_dict.items():
#     print(len(v[0]), type(v[0]), v[0])
    #exit()


# from accelerate import Accelerator
# from accelerate import FullyShardedDataParallelPlugin
# import inspect


# if training_config.get('distributed_type') == 'FSDP':
#     default_fsdp_parameters = inspect.signature(FullyShardedDataParallelPlugin.__init__).parameters
#     fsdp_args = {}
#     for param_name, param in default_fsdp_parameters.items():
#         config_key = f"fsdp_{param_name}"
#         if config_key in training_config['fsdp_config']:
#             fsdp_args[param_name] = training_config['fsdp_config'][config_key]
#         elif param.default is not inspect.Parameter.empty:
#             # If not assigned, use the default value
#             fsdp_args[param_name] = param.default
#     fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_args)
#     del training_config["fsdp_config"]
#     training_config["fsdp_plugin"] = fsdp_plugin

# transformer.wte 1 torch.Size([1, 2047, 4096])
# transformer.h.0.ln_1 1 torch.Size([1, 2047, 4096])

# for k,v in self.larger_hook_forward_dict.items():
#     print(k, v[0].shape)


# print(torch.equal(self.larger_hook_forward_dict["transformer.h.31.ln_2"],self.larger_hook_forward_dict["transformer.h.31.ln_1"]))
    
# import re
# pattern = r"transformer\.h\.\d+.ln_1$"
# for k,v in self.larger_hook_forward_dict.items():
#     if re.match(pattern, k):
#         print(k, v[0].shape)
#         print("-------")
#     else:
#         pass



# print(type(larger_hidden_states[0]))
# print(type(self.larger_hook_forward_dict["transformer.h.0.ln_1"]))

# print(larger_hidden_states[0].shape)
# print(self.larger_hook_forward_dict["transformer.h.0.ln_1"][0].shape)
# print(len(self.larger_hook_forward_dict["transformer.h.0.ln_1"]))
# #print(self.larger_hook_forward_dict["transformer.h.0.ln_1"][1][0].shape)
# #print(self.larger_hook_forward_dict["transformer.h.0.ln_1"][1][1].shape)

# print(torch.equal(self.larger_hook_forward_dict["transformer.h.0.ln_1"][0], larger_hidden_states[0]))

# print(torch.equal(self.larger_hook_forward_dict["transformer.h.0.ln_1"], larger_hidden_states[0]))
# print(torch.equal(self.larger_hook_forward_dict["transformer.h.0.ln_1"], larger_hidden_states[1]))

# print(torch.equal(self.larger_hook_forward_dict["transformer.h.31.ln_1"],larger_hidden_states[-1]))
# print(torch.equal(self.larger_hook_forward_dict["transformer.h.31.ln_1"],larger_hidden_states[-1]))

# for k,v in self.larger_hook_forward_dict.items():
#     print(k, len(v))

# print("=======")

# for idx in range(32):
#     #n = "transformer.h."+str(idx)+".ln_1"
#     #n = "transformer.h."+str(idx)
#     n = "transformer.h."+str(idx)+".attn"
#     print(idx, torch.equal(self.larger_hook_forward_dict[n][0], larger_hidden_states[idx]))

# # for k,v in self.larger_hook_forward_dict.items():
# #     print(k, len(v))


# for module_name, module in model.named_modules():
#     #hook = module.register_forward_hook(self._forward_hook(module_name, model_name, is_modifiy))
#     print(module_name, module)
#     #hook = module.register_forward_pre_hook(self.forward_pre_hook(module_name, model_name, is_modifiy))
#     #total_hook_list.append(hook)


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