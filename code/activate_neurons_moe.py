from transformers import AutoModel, MixtralModel
import torch


# neurons: batch_size x length x n_layer x n_experts x n_hidden
# moe_weights: batch_size x length x n_layer x n_experts
def activate_neurons(model, input_ids, neurons_device='cpu'):
    def func_n_layers_n_experts(model):
        if isinstance(model, MixtralModel):
            return len(model.layers), len(model.layers[0].block_sparse_moe.experts)
        raise ValueError(f'Unsupported model class {model.__class__.__name__}')

    def func_act_module(model, i_layer, i_expert):
        if isinstance(model, MixtralModel):
            return model.layers[i_layer].block_sparse_moe.experts[i_expert].act_fn
        raise ValueError(f'Unsupported model class {model.__class__.__name__}')

    def func_moe_block(model, i_layer):
        if isinstance(model, MixtralModel):
            return model.layers[i_layer].block_sparse_moe

    n_layers, n_experts = func_n_layers_n_experts(model)
    neurons = torch.zeros((input_ids.numel(), n_layers, n_experts, model.config.intermediate_size), dtype=model.config.torch_dtype, device=neurons_device)
    moe_weights = torch.zeros((input_ids.numel(), n_layers, n_experts), dtype=torch.float, device=neurons_device)

    def get_hook_func(i_layer, topk):
        expert_mask = None
        routing_weights = None

        def router_hook(module, input, output):
            nonlocal expert_mask, routing_weights
            router_logits = output
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_expert = torch.topk(routing_weights, topk, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(neurons_device)
            expert_mask = torch.nn.functional.one_hot(selected_expert, num_classes=n_experts).permute(2, 1, 0).to(neurons_device)

        def get_act_hook(i_layer, j):
            def act_hook(module, input, output):
                input = input[0].detach().to(neurons_device)
                idx, top_x = torch.where(expert_mask[j])
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()
                neurons[top_x_list, i_layer, j] = input
                moe_weights[top_x_list, i_layer, j] = routing_weights[top_x_list, idx_list]
            return act_hook

        act_hook_funcs = []
        for j in range(n_experts):
            act_hook_funcs.append(get_act_hook(i_layer, j))

        return router_hook, act_hook_funcs

    for i in range(n_layers):
        moe = func_moe_block(model, i)
        router_hook, act_hook_funcs = get_hook_func(i, moe.top_k)
        moe.gate.register_forward_hook(router_hook)

        for j, act_hook_func in enumerate(act_hook_funcs):
            func_act_module(model, i, j).register_forward_hook(act_hook_func)

    model(input_ids)

    neurons = neurons.reshape(*input_ids.shape, *neurons.shape[1:])
    moe_weights = moe_weights.reshape(*input_ids.shape, *moe_weights.shape[1:])

    return neurons, moe_weights


if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_name = 'mistralai/Mixtral-8x7B-v0.1'
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name)

    tok.pad_token = tok.eos_token

    input_ids = tok(['I have an apple.', 'I extremely love my dog, and also my cat.'], padding=True, return_tensors='pt').input_ids.cuda()
    neurons, moe_weights = activate_neurons(model, input_ids)
    print(neurons.dtype, neurons.shape, moe_weights.shape, neurons[moe_weights == 0].sum())
