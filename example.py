from tartarus import pce
from tartarus import tadf
from tartarus import docking
from tartarus import reactivity
from time import perf_counter
from genetic_gfn.scoring_function import sigmoid_transformation

# test and print all objecties
# dipm, gap, lumo, combined, 
# start = perf_counter()
# pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties('c1sc(-c2[SiH2]c(cc2)-c2ccc(-c3scc4occc34)c3cscc23)c2Cccc12')
# print('******* PCE *******')
# print(f'Evaluation time: {perf_counter() - start}')
# print(f'PCE1: {pce_pcbm_sas}')
# print(f'PCE2: {pce_pcdtbt_sas}')
# print()

smiles = 'O=C1NC2=C(O1)C1=C(N=CO1)C2=O'
for smiles in ['O=C1NC2=C(O1)C1=C(N=CO1)C2=O', 'C=CC=Cc1cc2oc(N)cc2[nH]1', 'C=CC=Cc1cc2[nH]c(=O)[nH]c2[nH]1']:
    start = perf_counter()
    st, osc, combined = tadf.get_properties(smiles)
    print('******* TADF *******')
    print(f'Evaluation time: {perf_counter() - start}')
    print(f'Singlet-triplet: {st}')
    print(f'Oscillator strength: {osc}')
    print(f'Combined obj: {combined}')
    print([sigmoid_transformation(osc, 'tadf_osc'), st, osc, combined])
    print()


# score = docking.perform_calc_single('Cl/C(=C/Cl)/CC(N)C(=O)O', '1syh', docking_program='qvina')
score = docking.perform_calc_single('NC1=CC(c2ccccc2)=N/C(=N/C2c3ccccc3Nc3ccccc32)N1', '6y2f', docking_program='qvina')
print('******* Docking *******')
print(f'Docking score: {score}')
print()

# Ea, Er, sum_Ea_Er, diff_Ea_Er  = reactivity.get_properties('CC=CC(C)=CC=CC=CC1CC2CC1C13CC21C1C=CC3C1', 
#     n_procs=1)  # set number of processes
# print('******* Reactivity *******')
# print(f'Activation energy: {Ea}')
# print(f'Reaction energy: {Er}')
# print(f'Activation + reactivity: {sum_Ea_Er}')
# print(f'Reactivity - activation: {diff_Ea_Er}')
# print()