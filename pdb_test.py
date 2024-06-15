"""exec(open('../pdb_test.py').read())"""

"""Write the code that you want to test here"""

print("\n")
print(f"==Test==")
print("\n")

print("!!!!!")
print(len(output), len(modified_output))
print("!!!!!")
#exit()

for idx in range(0, len(output)):
    if len(output[idx]) == 1:
        print(1111)
        print(output[idx].shape)
    else:
        print(2222)
        print(len(output[idx]), type(output[idx]))
        for idxx in range(0, len(output[idx])):
            if len(output[idx][idxx]) == 1:
                print(3333)
                print(output[idx][idxx].shape)
            else:
                print(44444)
                for idxxx in range(0, len(output[idxx])):
                    print(output[idx][idxx][idxxx].shape)

print("==============")

for idx in range(0, len(modified_output)):
    if len(modified_output[idx]) == 1:
        print(1111)
        print(modified_output[idx].shape)
    else:
        print(2222)
        print(len(modified_output[idx]), type(modified_output[idx]))
        for idxx in range(0, len(modified_output[idx])):
            if len(modified_output[idx][idxx]) == 1:
                print(3333)
                print(modified_output[idx][idxx].shape)
            else:
                print(44444)
                for idxxx in range(0, len(modified_output[idxx])):
                    print(modified_output[idx][idxx][idxxx].shape)


