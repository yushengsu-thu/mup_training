"""exec(open('../pdb_test.py').read())"""

"""Write the code that you want to test here"""

print("\n")
print(f"==Test==")
print("\n")


for idx in range(0, len(target_input)): 
    if len(target_input[idx]) == 1:
        print(111111)
        print(type(target_input[idx]), target_input[idx].shape)
    else:
        print(222222222)
        for i in range(0, len(target_input[idx])):
            print(type(target_input[idx][i]), target_input[idx][i].shape)
print("===================")
for idx in range(0, len(input)): 
    if len(input[idx]) == 1:
        print(111111)
        print(type(input[idx]), input[idx].shape)
    else:
        print(222222222)
        for i in range(0, len(input[idx])):
            print(type(input[idx][i]), input[idx][i].shape)

