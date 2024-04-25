import os
import glob
import numpy as np
# from tdc import Oracle, Evaluator

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import pybel

import subprocess
import multiprocessing

from tartarus import pce, tadf, docking, reactivity

TADF_INVALID_SCORE = -10**4
DOCKING_INVALID_SCORE = 10**4


# def int_div(smiles):
#     evaluator = Evaluator(name = 'Diversity')
#     return evaluator(smiles)

# def get_scores(smiles, mode="QED"):
#     smiles_groups = []
#     group_size = len(smiles) / n_process
#     for i in range(n_process):
#         smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

#     temp_data = []
#     # pool = multiprocessing.Pool(processes = n_process)
#     # for index in range(n_process):
#     #     temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, )))
#     # pool.close()
#     # pool.join()
#     # scores = []
#     # for index in range(n_process):
#     #     scores += temp_data[index].get()
#     scores = get_scores_subproc(smiles_groups[0], mode)

#     for filename in glob.glob("docking/mols/*"):
#         if os.path.exists(filename):
#             os.remove(filename)

#     return scores


def get_scores(smiles, mode="tadf_osc", n_process=16):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    pool = multiprocessing.Pool(processes = n_process)
    for index in range(n_process):
        temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, )))
    pool.close()
    pool.join()
    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()

    return scores

def get_scores_subproc(smiles, mode):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    # oracle_QED = Oracle(name='QED')
    # oracle_SA = Oracle(name='SA')

    if mode == "tadf_st":
        for i in range(len(smiles)):
            if mols[i] != None:
                st, osc, combined = tadf.get_properties(smiles[i])
                scores += [sigmoid_transformation(st, mode), st, osc, combined]
            else:
                scores += [-1.0, -1.0, -1.0, -1.0]

    elif mode == "tadf_osc":
        for i in range(len(smiles)):
            if mols[i] != None:
                st, osc, combined = tadf.get_properties(smiles[i])
                scores += [sigmoid_transformation(osc, mode), st, osc, combined]
            else:
                scores += [-1.0, -1.0, -1.0, -1.0]
    
    elif mode == "tadf_combined":
        for i in range(len(smiles)):
            if mols[i] != None:
                st, osc, combined = tadf.get_properties(smiles[i])
                import pdb; pdb.set_trace()
                scores += [sigmoid_transformation(combined, mode), st, osc, combined]
            else:
                scores += [-1.0, -1.0, -1.0, -1.0]
    
    elif mode.startswith('docking'):
        _, receptor, program = mode.split('_')
        for i in range(len(smiles)):
            if mols[i] != None:
                unnormalized = docking.perform_calc_single(smiles[i], receptor_type=receptor, docking_program=program)
                scores += [sigmoid_transformation(unnormalized, mode), unnormalized]
            else:
                scores += [-1.0, -1.0]

    elif mode == "docking_PLPro_7JIR_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                # docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                docking_score, unnormalized = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55], return_raw=True)
                qed = oracle_QED(smiles[i])
                sa = oracle_SA(smiles[i])  #(10 - oracle_SA(smiles[i])) / 9
                scores += [0.8 * docking_score + 0.1 * qed + 0.1 * (10 - sa) / 9, unnormalized, qed, sa]
                try:
                    ligand_mol_file = f"./docking/tmp/mol_{smiles[i]}.mol"
                    ligand_pdbqt_file = f"./docking/tmp/mol_{smiles[i]}.pdbqt"
                    docking_pdbqt_file = f"./docking/tmp/dock_{smiles[i]}.pdbqt"
                    for filename in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
                        if os.path.exists(filename):
                            os.remove(filename)
                except:
                    pass
                
                if docking_score > 0.8:
                    print(smiles[i], docking_score)
            else:
                scores += [-1.0, -1.0, 0.0, 0.0]

    elif mode == "docking_RdRp":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[93.88, 83.08, 97.29])
                scores += [docking_score]
            else:
                scores += [-1.0]

    elif mode == "docking_RdRp_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score, unnormalized = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[51.51, 32.16, -0.55], return_raw=True)
                qed = oracle_QED(smiles[i])
                sa = oracle_SA(smiles[i])  #(10 - oracle_SA(smiles[i])) / 9
                scores += [0.8 * docking_score + 0.1 * qed + 0.1 * (10 - sa) / 9, unnormalized, qed, sa]
                try:
                    ligand_mol_file = f"./docking/tmp/mol_{smiles[i]}.mol"
                    ligand_pdbqt_file = f"./docking/tmp/mol_{smiles[i]}.pdbqt"
                    docking_pdbqt_file = f"./docking/tmp/dock_{smiles[i]}.pdbqt"
                    for filename in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
                        if os.path.exists(filename):
                            os.remove(filename)
                except:
                    pass
                
                if docking_score > 0.8:
                    print(smiles[i], docking_score)
            else:
                scores += [-1.0, -1.0, 0.0, 0.0]

    else:
        raise Exception("Scoring function undefined!")

    return scores


def sigmoid_transformation(original_score, mode):
    if mode == "tadf_osc":
        _low = 0.
        _high = 3.
        _k = 0.25
        minimize = False
    elif mode.startswith("docking"):
        _low = -12
        _high = -8
        _k = 0.25
        minimize = True


    if original_score > 99.9 or original_score < -99.9:
        return -1.0

    def _sigmoid_formula(value, low, high, k) -> float:
        try:
            return 1 / (1 + 10 ** (-k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
        
    def _reverse_sigmoid_formula(value, low, high, k) -> float:
        try:
            return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
        
    if minimize:
        transformed = _reverse_sigmoid_formula(original_score, _low, _high, _k)
    else:
        transformed = _sigmoid_formula(original_score, _low, _high, _k)
    return transformed