# FF, BF hook: https://www.cnblogs.com/yh-blog/p/15564581.html#:~:text=hook%E6%98%AF%E9%92%A9%E5%AD%90%EF%BC%8C%E4%B8%BB%E8%A6%81%E4%BD%9C%E7%94%A8,%E5%8F%98%E9%87%8F%E7%9A%84%E5%80%BC%E5%92%8C%E6%A2%AF%E5%BA%A6%E3%80%82

from transformers import AutoModel, AutoTokenizer
import torch
import sys
#from pynvml import *
from datasets import load_dataset
from torch.utils.data import DataLoader
import os


'''
def activate_neurons(model, input_ids, neurons_device='cpu'):
    def func_n_layers(model):
        model_name = model.__class__.__name__
        if 'LlamaModel' in model_name:
            return len(model.layers)
        elif 'CrystalCoderModel' in model_name:
            return len(model.h)
        elif 'EleutherAI/pythia' in model_name or 'GPTNeoXModel' in model_name:
            return len(model.layers)
        raise ValueError(f'Unsupported model class {model_name}')

    def func_act_module(model, i_layer):
        model_name = model.__class__.__name__
        if 'LlamaModel' in model_name:
            return model.layers[i_layer].mlp.act_fn
        elif 'CrystalCoderModel' in model_name:
            return model.h[i_layer].mlp.act
        elif 'EleutherAI/pythia' in model_name or 'GPTNeoXModel' in model_name:
            return model.layers[i_layer].mlp.act
        raise ValueError(f'Unsupported model class {model_name}')

    neurons = []

    def get_hook_func(i):
        def func(module, input, output):
            input = input[0].detach().to(neurons_device)
            print(type(neurons))
            neurons.append(input)
        return func

    for i in range(func_n_layers(model)):
        func_act_module(model, i).register_forward_hook(get_hook_func(i))

    model(input_ids)


    #torch.Size([batch_size, layers, sentence_length, hidden_state_dim])
    #neurons = torch.stack(neurons, dim=2)
    neurons = torch.stack(neurons, dim=1)

    return neurons
'''



class NeuronActivator:
    def __init__(self, model, neurons_device='cpu'):
        self.model = model
        self.neurons = []
        self.neurons_device = neurons_device

    def func_n_layers(self):
        model_name = self.model.__class__.__name__
        if 'LlamaModel' in model_name:
            return len(self.model.layers)
        elif 'CrystalCoderModel' in model_name:
            return len(self.model.h)
        elif 'EleutherAI/pythia' in model_name or 'GPTNeoXModel' in model_name:
            return len(self.model.layers)
        raise ValueError(f'Unsupported model class {model_name}')

    def func_act_module(self, i_layer):
        model_name = self.model.__class__.__name__
        if 'LlamaModel' in model_name:
            return self.model.layers[i_layer].mlp.act_fn
        elif 'CrystalCoderModel' in model_name:
            return self.model.h[i_layer].mlp.act
        elif 'EleutherAI/pythia' in model_name or 'GPTNeoXModel' in model_name:
            return self.model.layers[i_layer].mlp.act
        raise ValueError(f'Unsupported model class {model_name}')

    def get_hook_func(self, i):
        def func(module, input, output):
            input = input[0].detach().to(self.neurons_device)
            self.neurons.append(input)
        return func

    def activate_neurons(self, input_ids):
        self.neurons = []
        for i in range(self.func_n_layers()):
            self.func_act_module(i).register_forward_hook(self.get_hook_func(i))

        self.model(input_ids)
        neurons_tensor = torch.stack(self.neurons, dim=1)
        return neurons_tensor




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = os.getcwd()+"/../cache"
    llm_name = sys.argv[1]

    dataset = load_dataset("Open-Orca/SlimOrca", cache_dir=target)
    '''
    dataloader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=False,
                        num_workers=64)
    for i, (x, y) in enumerate(dataloader):
    '''

    #model_name = 'LLM360/Amber'
    #model_name = 'LLM360/CrystalCoder'
    #model_name = 'openlm-research/open_llama_3b'
    model_name = 'EleutherAI/'+str(llm_name)

    #model = AutoModel.from_pretrainedm_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, cache_dir=target).to(device)
    token = AutoTokenizer.from_pretrained(model_name)

    token.pad_token = token.eos_token


    list_std = list()
    list_mean = list()
    all_input_ids = list()
    for idx, data in enumerate(dataset["train"]):

        if idx == 100:
            break

        input_ = data['conversations'][1]['value']


        #input_ids = token(['I have an apple.', 'I extremely love my dog, and also my cat.'], padding=True, return_tensors='pt').input_ids.to(device)
        input_ids = token([input_], padding=True, return_tensors='pt').input_ids.to(device)


        #torch.Size([batch_size, layers, sentence_length, hidden_state_dim])
        #ret = activate_neurons(model, input_ids)
        #ret = activate_neurons(model, input_ids)
        activator = NeuronActivator(model, 'cpu')
        ret = activator.activate_neurons(input_ids).to('cpu')
        ret = ret.reshape(ret.shape[1], ret.shape[2]*ret.shape[3])

        ret = torch.std_mean(ret, dim=1, keepdim=False)

        #print(ret[0].shape) # std
        #print(ret[1].shape) # mean

        list_std.append(ret[0])
        list_mean.append(ret[1])


    list_std = torch.stack(list_std, dim=0)
    list_mean = torch.stack(list_mean, dim=0)

    #model_name = 'EleutherAI/pythia-70m'
    _, name = model_name.split("/")

    target_dir = os.getcwd()+"/../neurons/"+str(name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        pass

    torch.save(list_std, str(target_dir)+'/list_std.pt')
    torch.save(list_mean, str(target_dir)+'/list_mean.pt')


