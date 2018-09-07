#!/usr/bin/env bash

conda update conda
conda create --yes -n python3 python=3.5
source activate python3
pip install --upgrade pip


pip install editdistance
pip install xlrd
pip install joblib
conda install --yes pandas numpy seaborn scikit-learn tqdm
pip install --upgrade gensim
pip install biopython
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.9.0-cp35-cp35m-linux_x86_64.whl

pip install "dask[complete]"

pip install keras

# DO NOT FORGET TO ADD RIKEN INSTALLATION PATH TO YOUR BASHRC !!


