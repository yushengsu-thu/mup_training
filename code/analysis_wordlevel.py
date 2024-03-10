import torch
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np




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
    if os.path.isfile(path+name_i+"/list_mean.pt"):
        pt_file = torch.load(path+name_i+"/list_mean.pt")
        print(name_i, pt_file.shape)
'''



# Generate three sets of data for X1 and X2, all within the range of 0 to 10
x1_series = np.random.randint(0, 11, size=(3, 11))  # Three series for X1
x2_series = np.random.randint(0, 11, size=(3, 11))  # Three series for X2

# Define bin edges for alignment
bins = np.arange(0, 12) - 0.5

# Create histograms for the three series of X1 and X2
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Loop through each series and plot only X1 and X2 with different hatching for differentiation
for i in range(3):
    axs[i].hist(x1_series[i], bins=bins, color='skyblue', alpha=0.7, label='X1 (Series {})'.format(i+1), rwidth=0.8, hatch='//')
    axs[i].hist(x2_series[i], bins=bins, color='skyblue', alpha=0.7, label='X2 (Series {})'.format(i+1), rwidth=0.8, hatch='\\\\')
    axs[i].legend()
    axs[i].set_ylabel('Frequency')

# Common settings
axs[-1].set_xticks(np.arange(0, 11))  # Set x-ticks to be at integer values
axs[-1].set_xlabel('Value')
fig.suptitle('Aligned Histograms for X1 and X2 Across 3 Series with Different Hatching')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
#plt.show()
target_dirname = f'../visual/img.pdf'
plt.savefig(target_dirname, format="pdf", bbox_inches="tight")

exit()



for idx_i in range(len(dir_list)):
    for idx_j in range(idx_i+1, len(dir_list)):
        name_i = dir_list[idx_i]
        name_j = dir_list[idx_j]

        name_i = path + name_i
        name_j = path + name_j
        if os.path.isdir(name_i) and os.path.isdir(name_j):
            length_ = os.listdir(name_i)

        #for idx in range(length_):
        for idx in range(2):
            #mean
            name_i_mean = f'{name_i}/{idx}_list_mean.pt'
            name_j_mean = f'{name_j}/{idx}_list_mean.pt'
            if os.path.isfile(name_i_mean):
                pt_i = torch.load(name_i_mean)
            else:
                break
            if os.path.isfile(name_j_mean):
                pt_j = torch.load(name_j_mean)
            else:
                break



        exit()


        '''
        i##mean
        #name_i_mean = path + name_i + "/list_mean.pt"
        #name_j_mean = path + name_j + "/list_mean.pt"

        if os.path.isfile(name_i_mean):
            pt_i = torch.load(name_i_mean)
        else:
            break
        if os.path.isfile(name_j_mean):
            pt_j = torch.load(name_j_mean)
        else:
            break
        '''
        print("=========================")
        print(name_i, ":", pt_i.shape)
        print(name_j, ":", pt_j.shape)
        scale_ratio = int(pt_j.shape[-1]/pt_i.shape[-1])
        print("scale_ratio:", scale_ratio)
        #dict_ = dict()
        print("----------mean-----------")
        for id_ in range(int(pt_j.shape[-1])):
            kld = F.pairwise_distance(pt_i[:,id_//scale_ratio], pt_j[:,id_])
            print("Layer:", id_, "Dis_mean:", float(kld))
            #print(float(kld))
        #print("=========================")


        #std
        name_i_std = path + name_i + "/list_std.pt"
        name_j_std = path + name_j + "/list_std.pt"
        if os.path.isfile(name_i_std):
            pt_i = torch.load(name_i_std)
        else:
            break
        if os.path.isfile(name_j_std):
            pt_j = torch.load(name_j_std)
        else:
            break
        #print("=========================")
        #print(name_i, ":", pt_i.shape)
        #print(name_j, ":", pt_j.shape)
        #scale_ratio = int(pt_j.shape[-1]/pt_i.shape[-1])
        #print("scale_ratio:", scale_ratio)
        #dict_ = dict()
        print("----------std-----------")
        for id_ in range(int(pt_j.shape[-1])):
            std = F.pairwise_distance(pt_i[:,id_//scale_ratio], pt_j[:,id_])
            print("Layer:", id_, "Dis_std:", float(std))
            #print(float(std))
        print("=========================")



#n_layer_neurons = torch.load("./myfile.pt")
