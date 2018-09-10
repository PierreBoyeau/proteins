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

## III. Install
Everything described here has been done with Ubuntu 16.06. 

1. Copy or clone this project via Github using:
    ```bash
    git clone https://github.com/PierreBoyeau/riken.git
    ```

2. Run `install.sh`. In this script we used extensively Anaconda (which you can install [here](https://www.anaconda.com/download)) but can easily be done manually using pip.
Just ensure that you use Python 3! 

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

- To train a model with specified parameters, save it and evaluate model performance on test data:
```bash
python rnn_keras_with_psiblast.py -data_path ~/riken/data/riken_data/complete_from_xlsx_v2COMPLETE.tsv \
-pssm_format_file ~/riken/data/psiblast/riken_data_v2/{}_pssm.txt \
-key_to_predict is_allergenic \
-index_col 0 \
-groups predefined
```


To have more information on what each parameter is used for, just type `python rnn_keras_with_psiblast.py --help`


- `rnn_cross_validation.py` is used to establish RNN model performance using a 10-fold species-specific cross-validation criteria.


- You can use `model_psiblast_eval.ipynb` to dynamically study an already trained model (using `rnn_keras_with_psiblast.py`).
In this notebook, you can plot ROC curves and visualize attention weights, in order to extract allergens motifs.


### B. Use cloud computation
- If you use NIT servers (griffin,fox, etc.), just execute directly the command in bash.
Just note that you should use griffin server (the only one with GPU support)

- If you use RIKEN raiden, note that you cannot directly execute the command. You must **submit**
you task to the server using a script like `server/script_example.sh`

Please note you can find doc about how riken servers work in `riken/doc`

### C. Compute PSSM matrices
As you understood, using RNN/TCN models assume you have at disposal PSIBLAST PSSMs matrice representation of your protein data.
If you work with new data, or if you want to redo the computation, you will have to have a look at 
`riken/protein_io/compute_pssm.py` using the following command:

```bash
python compute_pssm.py -data_path PATH_TO_PROTEINS.csv \
--save_dir DIRECTORY_WHERE_TO_SAVE_PSSM
```

As usual, run `python compute_pssm.py --help` for more guidelines.

### D. Older works
First experiments (Word2Vec/Allerdictor based) all are contained in `riken/word2vec`.
The most important file is `mdl_cross_validation.py`. This file allows you to get the **allerdictor** (`mode==svm`)
and **Word2Vec**(`mode=word2vec`) cross-validated performance