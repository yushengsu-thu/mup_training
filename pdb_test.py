"""exec(open('../pdb_test.py').read())"""

"""Write the code that you want to test here"""

print(f"==Test==\n")

for (name_f, param_f), (name_p, param_p) in zip(self.larger_hook_backward_dict.items(), self.smaller_hook_backward_dict.items()):
    if len(param_p) == 1 and len(param_f) == 1:
        #print(name_f, param_p[0].shape, param_f[0].shape)
        if param_p[0].shape == param_f[0].shape:
            print(name_f)


