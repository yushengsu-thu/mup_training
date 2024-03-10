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



'''
# Generating sample data for d1 and d2
np.random.seed(0)  # for reproducibility
d1 = np.random.uniform(-1, 1, 100)
d2 = np.random.uniform(-1, 1, 100)

# Plotting histograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(d1, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of d1')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(d2, bins=20, color='salmon', edgecolor='black')
plt.title('Histogram of d2')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
target_dirname = f'../visual/img.pdf'
plt.savefig(target_dirname, format="pdf", bbox_inches="tight")
exit()
'''



'''
#import numpy as np
#import matplotlib.pyplot as plt


sentence_length = 100
llm_layer = 6
layers = 3

# Generating 100 samples for each X1 and X2 in the specified ranges
#X1 = np.random.uniform(-1, 1, 100)
#X2 = np.random.uniform(-1, 1, 100)
X1 = np.random.uniform(-1, 1, size=(layers, sentence_length))  # Three series for X1
X2 = np.random.uniform(-1, 1, size=(layers, sentence_length))  # Three series for X2

for i in range(layers):
    plt.hist([X1[i], X2[i]], bins=20, alpha=0.5, label=['X1', 'X2'])

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of X1 and X2')
plt.legend()
#plt.show()

target_dirname = f'../visual/img.pdf'
plt.savefig(target_dirname, format="pdf", bbox_inches="tight")
exit()
'''



'''
sentence_length = 100
llm_layer = 6

# Generate three sets of data for X1 and X2, all within the range of -1 to 1
x1_series = np.random.uniform(-1, 1, size=(3, sentence_length))  # Three series for X1
x2_series = np.random.uniform(-1, 1, size=(3, sentence_length))  # Three series for X2
#print(x1_series.shape)

# Define bin edges for alignment
#bins = np.arange(0, 12) - 0.5
#bins = np.linspace(-1, 1, sentence_length+1)
bins = np.linspace(0, sentence_length, sentence_length+1)

# Create histograms for the three series of X1 and X2
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Loop through each series and plot only X1 and X2 with different hatching for differentiation
for i in range(3):
    #print(x1_series[i].shape)
    axs[i].hist(x1_series[i], bins=bins, color='skyblue', alpha=0.7, label='X1 (Series {})'.format(i+1), rwidth=0.8, hatch='//')
    axs[i].hist(x2_series[i], bins=bins, color='skyblue', alpha=0.7, label='X2 (Series {})'.format(i+1), rwidth=0.8, hatch='\\\\')
    axs[i].legend()
    axs[i].set_ylabel('Frequency')

# Common settings
#axs[-1].set_xticks(np.arange(0, sentence_length))  # Set x-ticks to be at integer values
axs[-1].set_xlabel('Value')
fig.suptitle('Aligned Histograms for X1 and X2 Across 3 Series with Different Hatching')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
#plt.show()


target_dirname = f'../visual/img.pdf'
plt.savefig(target_dirname, format="pdf", bbox_inches="tight")

exit()
'''



X = ['Group A','Group B','Group C','Group D']
Ygirls = [10,20,20,40]
Zboys = [20,30,25,30]

X_axis = np.arange(len(X))

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for ax in axs:
    ax.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls')
    ax.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys')

    ax.set_xticks(X_axis)
    ax.set_xticklabels(X)
    ax.set_xlabel("Groups")
    ax.set_ylabel("Number of Students")
    ax.set_title("Number of Students in each group")
    ax.legend()
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
        for idx in range(1):
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
