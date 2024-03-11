import torch
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import argparse




path = os.getcwd()+"/../neurons/wordlevel"
dir_list = os.listdir(path)
#print(dir_list)

#['pythia-70m', 'pythia-410m', 'pythia-1.4b', 'pythia-14m', 'pythia-160m', 'pythia-1b']
#dir_list = ["pythia-14m", 'pythia-70m','pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b']
'''
pythia-14m torch.Size([100, 6])
pythia-70m torch.Size([100, 6])
pythia-160m torch.Size([100, 12])
pythia-410m torch.Size([100, 24])
pythia-1b torch.Size([100, 16])
pythia-1.4b torch.Size([100, 24])
'''

#dir_list = ["pythia-14m", 'pythia-160m', 'pythia-410m', 'pythia-1.4b']
dir_list = ["pythia-14m", 'pythia-160m']

'''
for name_i in dir_list:
    print(path+name_i+"/0_neurons.pt")
    if os.path.isfile(path+"/"+name_i+"/0_neurons.pt"):
        pt_file = torch.load(path+"/"+name_i+"/0_neurons.pt")
        print(name_i, pt_file.shape)
exit()
'''

# 0.1 ~ 0.3: low
# 0.3 ~ 0.5: mid
# > 0.5: high
def pearson_correlation(x, y):
    # Ensure x and y have the same length
    assert x.size(0) == y.size(0), "Tensors must have the same size"

    # Calculate means
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    # Calculate the covariance between x and y
    covariance = torch.mean((x - x_mean) * (y - y_mean))

    # Calculate the standard deviations
    x_std = torch.sqrt(torch.var(x, unbiased=False))
    y_std = torch.sqrt(torch.var(y, unbiased=False))

    # Calculate the correlation coefficient
    correlation = covariance / (x_std * y_std)

    return correlation



def plot(tensor_1, tensor_2, name_1, name_2, sample_id):

    # torch.Size([layers, length, emb])
    tensor_1 = tensor_1.reshape(tensor_1.shape[0]*tensor_1.shape[1], tensor_1.shape[2], tensor_1.shape[3])
    #torch.Size([6, 39, 512])
    tensor_2 = tensor_2.reshape(tensor_2.shape[0]*tensor_2.shape[1], tensor_2.shape[2], tensor_2.shape[3])
    #torch.Size([12, 39, 3072])


    llm_layer = int(tensor_2.shape[0])
    rate = int(tensor_2.shape[0]/tensor_1.shape[0])
    sentence_length = int(tensor_1.shape[1])

    #X = ['Group A','Group B','Group C','Group D']
    X = [str(i) for i in range(sentence_length)]
    X_axis = np.arange(len(X))
    #mean
    tensor_1_mean = torch.mean(tensor_1, dim=-1)
    tensor_1_std = torch.std(tensor_1, dim=-1)
    #std
    tensor_2_mean = torch.mean(tensor_2, dim=-1)
    tensor_2_std = torch.std(tensor_2, dim=-1)

    fig_mean, axs_mean = plt.subplots(llm_layer, 1, figsize=(10, 15))
    #for layer_2 in range(llm_layer):
    for idx, ax in enumerate(axs_mean):
        layer_2 = idx
        layer_1 = layer_2//rate

        pc = pearson_correlation(tensor_1_mean[layer_1], tensor_2_mean[layer_2])
        tensor_1_mean_layer = tensor_1_mean[layer_1].tolist()
        tensor_2_mean_layer = tensor_2_mean[layer_2].tolist()
        ax.bar(X_axis - 0.15, tensor_1_mean_layer, 0.3, label = 'tensor_1')
        ax.bar(X_axis + 0.15, tensor_2_mean_layer, 0.3, label = 'tensor_2')

        ax.set_ylabel("Values")
        ax.set_title(f'{name_1}: layer {layer_1}, {name_2}: layer {layer_2}; Pearson Correlation: {pc}', fontsize=8)
        ax.legend(loc='right')
        if idx == len(axs_mean)-1:
            ax.set_xlabel("i_th tokens")
    target_dirname = f'../visual/{sample_id}_img_mean.pdf'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.savefig(target_dirname, format="pdf", bbox_inches="tight")



    fig_std, axs_std = plt.subplots(llm_layer, 1, figsize=(10, 15))
    #for layer_2 in range(llm_layer):
    for idx, ax in enumerate(axs_std):
        layer_2 = idx
        layer_1 = layer_2//rate
        pc = pearson_correlation(tensor_1_std[layer_1], tensor_2_std[layer_2])
        tensor_1_std_layer = tensor_1_std[layer_1].tolist()
        tensor_2_std_layer = tensor_2_std[layer_2].tolist()
        ax.bar(X_axis - 0.15, tensor_1_std_layer, 0.3, label = 'tensor_1')
        ax.bar(X_axis + 0.15, tensor_2_std_layer, 0.3, label = 'tensor_2')

        ax.set_ylabel("Values")
        #ax.set_title(f'{name_1}: layer {layer_1}; {name_2}: layer {layer_2}', fontsize=8)
        ax.set_title(f'{name_1}: layer {layer_1}, {name_2}: layer {layer_2}; Pearson Correlation: {pc}', fontsize=8)
        ax.legend(loc='right')
        if idx == len(axs_mean)-1:
            ax.set_xlabel("i_th tokens")

    target_dirname = f'../visual/{sample_id}_img_std.pdf'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.savefig(target_dirname, format="pdf", bbox_inches="tight")



def main(args):
    num_of_samples = args.num_of_samples

    for idx_i in range(len(dir_list)):
        for idx_j in range(idx_i+1, len(dir_list)):
            name_i_ = dir_list[idx_i]
            name_j_ = dir_list[idx_j]

            name_i = path + "/" + name_i_
            name_j = path + "/" + name_j_
            if os.path.isdir(name_i) and os.path.isdir(name_j):
                length_ = os.listdir(name_i)

            for idx in range(num_of_samples):
                #mean
                name_i_mean = f'{name_i}/{idx}_neurons.pt'
                name_j_mean = f'{name_j}/{idx}_neurons.pt'
                if os.path.isfile(name_i_mean):
                    pt_i = torch.load(name_i_mean)
                else:
                    break
                if os.path.isfile(name_j_mean):
                    pt_j = torch.load(name_j_mean)
                else:
                    break

                print(f'{idx}/{num_of_samples}')
                plot(pt_i, pt_j, name_i_, name_j_, idx)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_samples', type=int, default=10, help='num_of_samples')
    args = parser.parse_args()
    main(args)
