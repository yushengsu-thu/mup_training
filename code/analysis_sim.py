import torch
import os
import torch.nn.functional as F




path = os.getcwd()+"/../neurons/"
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

dir_list = ["pythia-14m", 'pythia-160m', 'pythia-410m', 'pythia-1.4b']

'''
for name_i in dir_list:
    if os.path.isfile(path+name_i+"/list_mean.pt"):
        pt_file = torch.load(path+name_i+"/list_mean.pt")
        print(name_i, pt_file.shape)
'''



for idx_i in range(len(dir_list)):
    for idx_j in range(idx_i+1, len(dir_list)):
        name_i = dir_list[idx_i]
        name_j = dir_list[idx_j]

        #mean
        name_i_mean = path + name_i + "/list_mean.pt"
        name_j_mean = path + name_j + "/list_mean.pt"
        if os.path.isfile(name_i_mean):
            pt_i = torch.load(name_i_mean)
        else:
            break
        if os.path.isfile(name_j_mean):
            pt_j = torch.load(name_j_mean)
        else:
            break
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
        print("=========================")


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
            kld = F.pairwise_distance(pt_i[:,id_//scale_ratio], pt_j[:,id_])
            print("Layer:", id_, "Dis_std:", float(kld))
        print("=========================")



#n_layer_neurons = torch.load("./myfile.pt")
