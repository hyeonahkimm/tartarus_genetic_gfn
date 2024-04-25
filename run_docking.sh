#!/bin/bash 

oracle_array=('docking_1syh_qvina', 'docking_1syh_smina',\
                'docking_6y2f_qvina', 'docking_6y2f_smina',\
                'docking_4lde_qvina', 'docking_4lde_smina')

for seed in 1 2 3 4
do
for oralce in "${oracle_array[@]}"
do
python train.py genetic_gfn --batch_size 400 --population_size 400 --offspring_size 50 --wandb online --oracle $oralce --seed $seed
done
done