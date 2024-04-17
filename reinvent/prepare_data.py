import os
import pandas as pd

data_path = 'data'
filename = 'hce.txt'
sep = ' '
header = None
smile_name = 0

# dataset load
fname = os.path.join(data_path, filename)
data = pd.read_csv(fname, sep=sep, header=header)
smiles = data[smile_name]

# create smi file
with open(os.path.join(data_path, 'data.smi'), 'w') as f:
    for smi in smiles:
        f.write(smi+'\n')
