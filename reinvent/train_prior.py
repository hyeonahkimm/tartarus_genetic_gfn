#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
from custom import EarlyStopping
rdBase.DisableLog('rdApp.error')

parser = argparse.ArgumentParser(description="Pretraining the model.")
parser.add_argument('--num-epochs', action='store', dest='num_epochs', type=int, 
     default=100, help='Number of epochs for pretraining.')
parser.add_argument('--train-ratio', action='store', dest='train_ratio', type=float,
     default=0.8, help='Ratio of data used for training.')
parser.add_argument('--verbose', action='store_true', dest='verbose', default=False, 
    help='Toggle amount of printing. Turn on for progress bars.')


def pretrain(num_epochs, verbose, train_ratio, restore_from=None, save_dir='data/Prior.ckpt'):
    """Trains the Prior RNN"""

    # Initialize early stopper
    early_stop = EarlyStopping(patience=10, min_delta=1e-7, mode='minimize')

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./reinvent/data/Voc")

    # Create a Dataset from a SMILES file
    moldata = MolData("./reinvent/data/mols_filtered.smi", voc)
    train_size = int(len(moldata)*train_ratio)
    # train_size = int(100*train_ratio)
    train_set = torch.utils.data.Subset(moldata, range(0, train_size))
    valid_set = torch.utils.data.Subset(moldata, range(train_size, len(moldata)))

    # training and validation set
    train_data = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=MolData.collate_fn)
    valid_data = DataLoader(valid_set, batch_size=128, collate_fn=MolData.collate_fn)

    Prior = RNN(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, num_epochs + 1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        Prior.rnn.train()
        for step, batch in tqdm(enumerate(train_data), total=len(train_data), disable=not verbose):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 50 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.003)
                if verbose:
                    tqdm.write("*" * 50)
                    tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                else:
                    print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        if verbose:
                            tqdm.write(smile)
                        else:
                            print(smile)
                if verbose:
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")
                else:
                    print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                # torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        # validation loop
        Prior.rnn.eval()
        val_loss = 0
        for step, batch in tqdm(enumerate(valid_data), total=len(valid_data), disable=not verbose):
            with torch.no_grad():
                seqs = batch.long()
                log_p, _ = Prior.likelihood(seqs)
                val_loss += - log_p.mean()

        val_loss /= len(valid_data)
        stop = early_stop.check_criteria(Prior.rnn, epoch, val_loss.item())
        if stop:
            Prior.rnn = early_stop.restore_best(Prior.rnn)
            break

        Prior.rnn.train()

    # Save the Prior
    torch.save(Prior.rnn.state_dict(), save_dir)

if __name__ == "__main__":
    arg_dict = vars(parser.parse_args())
    pretrain(num_epochs=arg_dict['num_epochs'], verbose=arg_dict['verbose'], train_ratio=arg_dict['train_ratio'])
