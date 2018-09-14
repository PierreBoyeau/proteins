# Readme

## I. Presentation
This project aims to predict allergenicity based on protein sequences and properties.

## II. Project content

This part objective is to give an overview of how the project is structured
### 1) `riken` Python package:

- **nn_utils**: General functions and utilities useful to train neural networks using TensorFlow with protein-based data.
Most of it is not currently used in the Keras implementation, but it is very useful to create tensorflow *records* if you use very big datasets.
- **protein_io**: Read, manipulate and generate data. In this folder you will find all you need to read csv, split you data into train/test or do cross validation
while in the same time taking into account appartenance to species.
- **rnn**: Recurrent neural networks. They are implemented in 2 ways, Tensorflow (in `tf` folder) and Keras, which was used primarly.
The keras implementation allows you to perform Transfer Learning.
I also tried to implement an autoencoder (`keras_autoencoder.py`) for transfer learning purposes, but I did not have time to try it.

- **similarity_learning**: Similarity learning that was used to try for feature selection.
- **spp**: Experiments based on Safe Pattern Pruning
- **tcn**: Temporal Convolution Networks (organization and use is almost the same as **rnn**)
- **word2vec**: Allergen prediction based on Word2Vec features, inspired by the Allerdictor and the ProtVec papers.

### 2) `data` folder
#### Structure
Folders that contain protein datasets:
- **riken_data**: allergens/non-allergens contained in files `complete_from_xlsx_v2COMPLETE.tsv`
- **swiss**: data extracted from the SwissProt dataset with clan tags
- **pfam**: data extracted from PFAM with clans tags

**psiblast** folder does not contain protein sequences but **features**. For different datasets, one can find in this folder amino-acid specific features obtained using PSIBLAST.

#### How data is stored?

Protein data is not stored as fasta files are they usually are, but in simple csv files with the 
following properties:
- tab-separated
- amino-acids are contained in a `sequences` column.

### 3) Other folders
`misc` contains articles linked to worked done here, on proteins and on the different algorithms ideas that helped us. It also contains figures and some output data.

`spp_sequence_ver2` is a copy of the C++ code given to me by Takuto-san.

## IIIA. Install (RIKEN servers)
1. Copy or clone this project via Github using:
    ```bash
    git clone https://github.com/PierreBoyeau/riken.git
    ```
2. Unzip the `riken_data` PSIBLAST PSSM files. Supposing you are in source folder in bash console:
    ```bash
    cd ./data/psiblast
    zip -s 0 riken_data_v2.zip --out unsplit.zip
    unzip unsplit.zip
    ```


**THIS IS ALL YOU NEED TO DO, AS PACKAGES INSTALL SHOULD BE DONE IN THE SCRIPT YOU SUBMIT**

## IIIB. Install (not on RIKEN servers)
Everything described here has been done with Ubuntu 16.06. 

1. Copy or clone this project via Github using:
    ```bash
    git clone https://github.com/PierreBoyeau/riken.git
    ```
2. Unzip the `riken_data` PSIBLAST PSSM files. Supposing you are in source folder in bash console:
    ```bash
    cd ./data/psiblast
    zip -s 0 riken_data_v2.zip --out unsplit.zip
    unzip unsplit.zip
    ```


2. Run `install.sh`. In this script we used extensively Anaconda (which you can install [here](https://www.anaconda.com/download)) but can easily be done manually using pip.
Just ensure that you use Python 3! 

2. **Important!!**: Update your `PYTHONPATH` variable. On unix-based machines (like NIC/Riken servers), 
execute the following command:
        ```bash
        export PYTHONPATH="/home/YOUR_USER_NAME/riken:$PYTHONPATH"
        ```
where `YOUR_USER_NAME` is your name on the **machine that will execute the scripts** (e.g. RIKEN or NIC server).

3. (Optional) If you want to generate PSIBLAST features, you will need to install the BLAST+ standalone software. Go [there](https://www.ncbi.nlm.nih.gov/books/NBK52640/) for instructions.
## IV. Use cases

Following parts describe some use cases, in a random order, to help you get started and use this project.

### A. Recurrent Neural Networks (`riken/rnn` folder)
Most techniques described here can be done in the same way with TCN (Temporal Convolution Networks) that we also experimented but that did not show as much potential as RNN in the `riken`

Model architecture is implemented in `rnn_keras_with_psiblast.py` file.

This architecture takes two inputs:
1. amino-acids sequences. We extract features from this info. First, we extract One-Hot representation of amino-acids. 
We also get physico-chemical properties of amino-acids from several sources.

1. dynamic representation of amino-acids (`psiblast_prop`). We also need the PSIBLAST-PSSM representations of sequences to be computed and stored locally for RNN scripts to work efficiently.

More information on motivations and problem definition can be found in the memo.


Now, let's describe what you can do:

- **Classical training**: To train a model with specified parameters, save it and evaluate model performance on test data:
```bash
python rnn_keras_with_psiblast.py -data_path ~/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv \
-pssm_format_file ~/riken/data/psiblast/riken_data_v2/{}_pssm.txt \
-key_to_predict is_allergenic \
-index_col 0 \
-groups predefined \
-log_dir experiment1

```


To have more information on what each parameter is used for, just type `python rnn_keras_with_psiblast.py --help`

- **Transfer Learning**: 2018/08/27 meeting with biologists showed that transfer learning based on clan prediction
was not a good idea. That being said, transfer learning in an unsupervised fashion (using Autoencoder) for instance
can be interesting.
For such purposes, you can also use `nn_keras_with_psiblast.py` to do so. Let's imagine you have trained an autoencoder (LSTM based for instance). 
By selecting a meaningful layer output of the Autoencoder, you can apply transfer learning using the command:
```bash
python rnn_keras_with_psiblast.py -data_path ~/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv \
-pssm_format_file ~/riken/data/psiblast/riken_data_v2/{}_pssm.txt \
-key_to_predict is_allergenic \
-index_col 0 \
-groups predefined \
-log_dir autoencoder_folder \
-transfer_path autoencoder_folder/trained_autoencoder_weights.hdf5 \
-layer_name meaningful_layer_name
```

- **Cross-Validation**: `rnn_cross_validation.py` is used to establish RNN model performance using a 10-fold species-specific cross-validation criteria.


- **Trained Model performance and attention study**: You can use `model_psiblast_eval.ipynb` to dynamically study an already trained model (using `rnn_keras_with_psiblast.py`).
In this notebook, you can plot ROC curves and visualize attention weights, in order to extract allergens motifs.


### B. Use cloud computation
- If you use NIT servers (griffin,fox, etc.), just execute directly the command in bash.
Just note that you should use griffin server (the only one with GPU support)

- If you use RIKEN raiden, note that you cannot directly execute the command. You must **submit**
you task to the server using a script like `server/script_example.sh`. **Important**: please make sure
to replace `pierre` by your username in the line: `export PYTHONPATH="/home/pierre/riken:$PYTHONPATH"
`

- DON'T EXECUTE THIS SCRIPT directly in the console. To submit the script to the server, 
execute instead `qsub script_example.sh`. To see the status of your jobs, execute `qstat`.

    Refer to Riken RAIDEN doc for more info.

Please note you can find doc about how riken servers work in `riken/doc`

### C. Compute PSSM matrices
As you understood, using RNN/TCN models assume you have at disposal PSIBLAST PSSMs matrice representation of your protein data.

The first step is to download the Dataset file that will be used by PSIBLAST to compute alignments.
Don't worry, you only need to do that once.

1. Download SwissProt dataset (in PSIBLAST accepted format). Go there ftp://ftp.ncbi.nlm.nih.gov/blast/db/
and download `swissprot.tar.gz`

2. Extract downloaded archive in a folder contained in `data/psiblast/swissprot`. `ls data/psiblast/swissprot` should output something like:
```bash
swissprot.00.phr
swissprot.00.pin
swissprot.00.pnd
swissprot.00.pni
swissprot.00.pog
swissprot.00.ppd
...
```

Now **you can compute PSSM files!**

If you work with new data, or if you want to redo the computation of PSSM files, go to 
`riken/protein_io`  and use the following command:

```bash
python compute_pssm.py -data_path PATH_TO_PROTEINS.csv \
--save_dir DIRECTORY_WHERE_TO_SAVE_PSSM
```

As usual, run `python compute_pssm.py --help` for more guidelines.


**NB**: you can use other datasets that `swissprot` to make alignments. Have a look at available datasets located in 
ftp://ftp.ncbi.nlm.nih.gov/blast/db and download the archive of your choice and extract it in the same way 
as explained for `swissprot`.
Simply add argument `--db PATH_TO_DB` to `compute_pssm.py` and you are good to go.

### D. Older works
First experiments (Word2Vec/Allerdictor based) all are contained in `riken/word2vec`.
The most important file is `mdl_cross_validation.py`. 

This file allows you to get the **allerdictor** and **Word2Vec** cross-validated performance
To get allerdictor cross_validated results, change **MODE** value to be `'svm'` or **MODE** value to be 
`word2vec` in `mdl_cross_validation.py` code.



## V. Remarks and warnings

- To import data, this project assumed that this project folder was located in `/home/pierre`. If at some point
you have some `FileNotFoundError`, make sure that said path exist on your machine, and if not just modify the code.


## VI. Data source

If you need to download bigger datasets that were considered for transfer learning, please find paths here:

- SwissProt (aka UniProt): https://www.uniprot.org/uniprot/?query=reviewed:yes#

- PFAM: Huge dataset of proteins containing clan (aka superfamily tags) information. It can be found here
ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam31.0/ (or ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases, take the last version)