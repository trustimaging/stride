#!/usr/bin/env bash

# create a temp directory
mkdir tmp
cd tmp

# download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda.sh
bash anaconda.sh -f -b -p ~/anaconda3

# configure anaconda
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false

# remove temp directory
cd ..
rm -r tmp

# install stride
git clone https://github.com/trustimaging/stride.git
cd stride
conda env create -f environment.yml
conda activate stride
pip install -e .
