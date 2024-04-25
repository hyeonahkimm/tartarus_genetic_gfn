import numpy as np
import copy

import rdkit.Chem as Chem
from rdkit.Chem import Draw, Descriptors
from rdkit import RDLogger

from tartarus import pce

RDLogger.DisableLog("rdApp.*")

class tartarus_score():
    
    kwargs = ["task", "obj_idx", "invalid_score", "coefficient"]
    task = ''
    obj_idx = -1
    invalid_score = 0.
    coefficient = 1

    def __init__(self):
        if self.task == 'pce':
            from tartarus import pce
            # self.fitness_function = pce.get_surrogate_properties
            self.fitness_function = pce.get_properties
        elif self.task == 'tadf':
            from tartarus import tadf
            self.fitness_function = tadf.get_properties
        elif self.task.startswith('docking'):
            from tartarus import docking
            _, receptor, program = self.task.split('_')
            assert receptor in ['1syh', '6y2f', '6y2f'] and program in ['qvina', 'smina']
            self.fitness_function = docking.perform_calc_single
        elif self.task == 'reactivity':
            from tartarus import reactivity
            self.fitness_function = reactivity.get_properties
            
    def __call__(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return self.invalid_score
        except:
            return self.invalid_score
        if self.task.startswith('docking'):
            _, receptor, program = self.task.split('_')
            score = self.fitness_function(smi, receptor, docking_program=program)
            # if score < -100:
            #     score = 0
            # else:
            #     score *= -1
            return (-1) * score
        else:
            return self.fitness_function(smi)[self.obj_idx]


def fitness_function(smi):
    try:
        # pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smi)
        mol = Chem.MolFromSmiles(smi)
        log_P = Descriptors.MolLogP(mol)
        return log_P
    
    except:
        return None

class custom_score():

    def __init__(self):
        pass

    def __call__(self, smi):
        import pdb; pdb.set_trace()
        return fitness_function(smi)

class EarlyStopping():
    ''' Class that checks criteria for early stopping. Saves the best model weights.
    '''
    def __init__(self, patience, min_delta, mode='minimize'):
        self.patience = patience
        self.best_weights = None
        self.checkpoint = 0
        self.best_epoch = 0
        if mode == 'maximize':
            self.monitor_fn = lambda a, b: np.greater(a - min_delta, b)
            self.best_val = -np.inf
        elif mode == 'minimize':
            self.monitor_fn = lambda a, b: np.less(a + min_delta, b)
            self.best_val = np.inf
        else:
            raise ValueError(f'Mode should be either minimize or maximize.')

    def check_criteria(self, net, epoch, new_val):
        ''' Compare with value in memory. If there is an improvement, reset the checkpoint and
        save the model weights.
        Return True if stopping criteria is met (checkpoint is exceeds patience), otherwise, return False.
        '''
        if self.monitor_fn(new_val, self.best_val):
            self.best_val = new_val
            self.checkpoint = 0
            self.best_weights = copy.deepcopy(net.state_dict())
            self.best_epoch = epoch
        else:
            self.checkpoint += 1
        
        return self.checkpoint > self.patience

    def restore_best(self, net, verbose=True):
        if verbose: print(f'        Early stopping at epoch: {self.best_epoch}       loss: {self.best_val}')
        net.load_state_dict(self.best_weights)
        return net