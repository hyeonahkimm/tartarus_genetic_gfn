import os
import glob
import numpy as np
# from tdc import Oracle, Evaluator

from tqdm import tqdm
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

# def get_scores(smiles, mode="tadf_osc"):
#     scores = get_scores_subproc(smiles, mode)

#     return scores


def get_scores(smiles, mode="tadf_osc", n_process=16):
    if mode.startswith('tadf'):
        scores = get_scores_subproc(smiles, mode)
        return scores

    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    scores = get_scores_subproc(smiles_groups[0], mode)

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
        pbar = tqdm(range(len(smiles)))
        for i in pbar:
            if mols[i] != None:
                st, osc, combined = tadf.get_properties(smiles[i])
                scores += [sigmoid_transformation(osc, mode), st, osc, combined]
                # tqdm.write(f'{smiles[i]}: {scores[-4:]}')
                pbar.set_postfix({'score': scores[-4], 'osc': osc})
            else:
                scores += [-1.0, -1.0, -1.0, -1.0]
    
    elif mode == "tadf_combined":
        for i in range(len(smiles)):
            if mols[i] != None:
                st, osc, combined = tadf.get_properties(smiles[i])
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