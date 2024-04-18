import sys, os
import pandas as pd
sys.path.append('reinvent')

from data_structs import canonicalize_smiles_from_file, construct_vocabulary, write_smiles_to_file
from train_prior import pretrain
from train_agent import train_agent


if __name__ == "__main__":
    config = {'pce_pcbm_sas': 
              {
        'pretrain_dataset': 'hce.csv',
        'pretrain_ckpt': 'Prior_pce.ckpt',
        'score': 'pce',
        'obj_idx': 0,
        'invalid_score': -10,
        'coefficient': 1  # maximize
        },
        'tadf_osc': 
        {
        'pretrain_dataset': 'gdb13.csv',
        'pretrain_ckpt': 'Prior_tadf.ckpt',
        'score': 'tadf',
        'obj_idx': 1,
        'invalid_score': 0.,
        'coefficient': 1  # maximize
        },
        '6y2f_qvina': 
        {
        'pretrain_dataset': 'docking.csv',
        'pretrain_ckpt': 'Prior_docking.ckpt',
        'score': 'docking_6y2f_qvina',
        'obj_idx': -1,
        'invalid_score': -1000,
        'coefficient': -1  # minimize
        },

    }

    task_config = config['tadf_osc']

    data_path = os.path.join('.', 'datasets')
    filename = task_config['pretrain_dataset']  #'gdb13.csv'
    sep = ','
    header = 'infer'
    smile_name = 'smiles'

    # dataset load
    fname = os.path.join(data_path, filename)
    data = pd.read_csv(fname, sep=sep, header=header)
    smiles = data[smile_name]

    # if not os.path.isdir('reinvent/data'):
    #     os.mkdir('./reinvent/data')

    # # create smi file
    # with open(os.path.join('reinvent/data', 'data.smi'), 'w') as f:
    #     for smi in smiles:
    #         f.write(smi+'\n')

    # smiles_file = 'reinvent/data/data.smi'
    # print("Reading smiles...")
    # smiles_list = canonicalize_smiles_from_file(smiles_file)
    # print("Constructing vocabulary...")
    # voc_chars = construct_vocabulary(smiles_list)
    # write_smiles_to_file(smiles_list, "reinvent/data/mols_filtered.smi")
    
    num_epochs = 100
    verbose = False
    train_ratio = 0.8

    # import pdb; pdb.set_trace()

    # pretrain(num_epochs=num_epochs, verbose=verbose, train_ratio=train_ratio, save_dir='reinvent/data/'+task_config['pretrain_ckpt'])
    
    train_agent(
        restore_prior_from='reinvent/data/'+task_config['pretrain_ckpt'],
        restore_agent_from='reinvent/data/'+task_config['pretrain_ckpt'],
        scoring_function='tartarus_score',
        scoring_function_kwargs={'task':task_config['score'], 'obj_idx': task_config['obj_idx'], 'invalid_score': task_config['invalid_score']},
        batch_size = 500,
        n_steps = 10,
        num_processes = -1,
    )
    