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
    name_i = dir_list[idx_i]

    #mean
    name_i_mean = path + name_i + "/list_mean.pt"
    if os.path.isfile(name_i_mean):
        pt_i_mean = torch.load(name_i_mean)
    else:
        break

    #print(pt_i.shape)
    #exit()

    print("=========================")
    #print(name_i, ":", pt_i.shape)
    print(name_i)
    #print("scale_ratio:", scale_ratio)
    #dict_ = dict()
    print("----------mean-----------")



    #std
    name_i_std = path + name_i + "/list_std.pt"
    if os.path.isfile(name_i_std):
        pt_i_std = torch.load(name_i_std)
    else:
        break

    #print(pt_i_mean.shape)
    #exit()

    #m = float(torch.mean(pt_i_mean, 0, True))

    #m = mean(pt_i_mean)

    #s = float(torch.mean(pt_i_std, 0, True))

    #print(m.shape)
    #exit()

    print(pt_i_mean[0])
    print("----")
    print(pt_i_std[0])





#n_layer_neurons = torch.load("./myfile.pt")
