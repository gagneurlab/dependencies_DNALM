# dependencies_DNALM

Code repository for the manuscript: *Nucleotide dependency analysis of DNA language models reveals genomic functional elements*

## Description

This repository contains code for the manuscript and general code to compute and visualize nucleotide dependencies using DNA language models. 
**Please refer to the notebook `compute_and_visualize_dep_maps.ipynb` for a quick start** , it includes examples and code to:

- Visualize nucleotide dependency maps for a specific sequence and DNA Language Model
- Compute variant influence scores for a specific sequence and DNA Language Model

## Requirements and Installation

### Software
SpeciesLM and RiNALMo models require FlashAttention-2 to be installed (https://github.com/Dao-AILab/flash-attention). For details on the packages and versions we used during development and testing, please refer to requirements.txt. These recommendations are provided for reproducibility but your code may run with other versions. The software was developed and tested on Linux using Python 3.8.17. Installing the required packages and setting up the environment usually takes about 30–60 minutes, depending on your hardware.

### Hardware
NVIDIA GPU (tested on A40)

## Data

Data with intermediate files for the different manuscript notebooks can be found at: https://doi.org/10.5281/zenodo.14883091

## SpeciesLM availability

The SpeciesLM models are available in huggingface at https://huggingface.co/collections/johahi/specieslms-678a39261cfff01c1fa3ae41 or at https://doi.org/10.5281/zenodo.14883091. 