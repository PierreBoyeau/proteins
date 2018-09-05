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

## III. Use cases
