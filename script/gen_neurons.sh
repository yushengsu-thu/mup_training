
for VARIABLE in "pythia-14m" "pythia-70m" "pythia-160m" "pythia-410m" "pythia-1b" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b"
do
    echo $VARIABLE
    CUDA_VISIBLE_DEVICES=1 python3 ../code/activate_neurons_pythia.py $VARIABLE
done


'''
for VARIABLE in "pythia-14m"
do
    python3 ../code/activate_neurons_pythia.py $VARIABLE
done
'''

