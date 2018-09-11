#!/bin/bash
#$ -S /bin/bash
#$ -jc gpu-container_g1.72h
#$ -ac d=nvcr-tensorflow-1807-py3

# Above are the instructions given to the server (regarding hardware spec and container choice)
# Below is the package installation steps.
# Feel free to add installations steps if you need to
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

# Here write your script
cd /home/pierre/riken/riken/rnn
python rnn_keras_with_psiblast.py -data_path ~/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv \
-pssm_format_file ~/riken/data/psiblast/riken_data_v2/{}_pssm.txt \
-key_to_predict is_allergenic \
-index_col 0 \
-groups predefined