from transformers import AutoModel
import torch

def activate_neurons(model, input_ids, neurons_device='cpu'):
    def func_n_layers(model):
        model_name = model.__class__.__name__
        if model_name == 'LlamaModel':
            return len(model.layers)
        if model_name == 'CrystalCoderModel':
            return len(model.h)
        raise ValueError(f'Unsupported model class {model_name}')

    def func_act_module(model, i_layer):
        model_name = model.__class__.__name__
        if model_name == 'LlamaModel':
            return model.layers[i_layer].mlp.act_fn
        if model_name == 'CrystalCoderModel':
            return model.h[i_layer].mlp.act
        raise ValueError(f'Unsupported model class {model_name}')

    neurons = []

    def get_hook_func(i):
        def func(module, input, output):
            input = input[0].detach().to(neurons_device)
            neurons.append(input)
        return func

    for i in range(func_n_layers(model)):
        func_act_module(model, i).register_forward_hook(get_hook_func(i))

    model(input_ids)

    neurons = torch.stack(neurons, dim=2)
    return neurons

if __name__ == '__main__':
    from transformers import AutoTokenizer
    # model_name = 'LLM360/Amber'
    model_name = 'LLM360/CrystalCoder'
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name)

    tok.pad_token = tok.eos_token

    input_ids = tok(['I have an apple.', 'I extremely love my dog, and also my cat.'], padding=True, return_tensors='pt').input_ids.cuda()
    ret = activate_neurons(model, input_ids)
    print(ret.dtype, ret.shape)
