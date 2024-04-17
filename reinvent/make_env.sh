#/bin/bash
module load python/3.6 scipy-stack
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3


virtualenv --no-download ~/env/reinvent
source  ~/env/reinvent/bin/activate
pip install --upgrade pip

pip install torch
pip install scikit-learn
pip install tqdm
pip install pexpect
pip install seaborn

deactivate
