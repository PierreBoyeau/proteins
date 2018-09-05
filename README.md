# Readme

## I. Presentation
This project aims to predict allergenicity based on protein sequences and properties.
## II. Project content

This part objective is to give an overview of how the project is structured
### 1) `riken` Python package:

- **nn_utils**: General functions and utilities useful to train neural networks using TensorFlow with protein-based data
- **protein_io**: Read, manipulate and generate data
- **rnn**: Recurrent neural networks
- **similarity_learning**: Similarity learning
- **spp**: Experiments based on Safe Pattern Pruning
- **tcn**: Temporal Convolution Networks
- **word2vec**: Allergen prediction based on Word2Vec features

### 2) `data` folder
Folders that contain protein datasets:
- **riken_data**: allergens/non-allergens contained in files `complete_from_xlsx_v2COMPLETE.tsv`
- **swiss**: data extracted from the SwissProt dataset with clan tags
- **pfam**: data extracted from PFAM with clans tags

**psiblast** folder does not contain protein sequences but **features**. For different datasets, one can find in this folder amino-acid specific features obtained using PSIBLAST.

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

### A. Recurrent Neural Networks
Most techniques described here can be done in the same way with TCN (Temporal Convolution Networks) that we also experimented but that did not show as much potential as RNN in the `riken`
