#!/bin/bash
#$ -S /bin/bash
#$ -jc gpu-container_g1.168h
#$ -ac d=nvcr-tensorflow-1807-py3

. /fefs/opt/dgx/env_set/nvcr-tensorflow-1807-py3.sh

export PYTHONPATH="/home/pierre/riken:$PYTHONPATH"

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

pip install --upgrade --user pip
pip install joblib --user
pip install biopython --user
pip install pandas --user
pip install numpy --user --upgrade
pip install seaborn --user
pip install scikit-learn --user
pip install tqdm --user
pip install keras --upgrade --user

cd /home/pierre/riken/riken/tcn
python tcn_cross_validation.py
