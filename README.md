# Prediction of pKa from the concatenated vector
This is the code and data for the paper, "Feature selection in molecular graph neural networks based on quantum chemical approaches". This repository contains the code to predict pKa from the concatenated vector H<sup>(L)</sup>.
The concatenated vectors about the molecules listed in Opt1_acidic_tst.csv were stored in element_data.

## Usage

In the training and prediction, the concatenated vectors in ./element_data directory are used. 
The concatenated vectors (H<sup>(L)</sup>) in element_data were prepared under the following condistions:
   - number of convolution layers (LAYER) : 1, 2, 3, 5, 7, or 10
   - atomic feature (TYPE): MAP_CAM (CAM-B3LYP/aug-cc-pVDZ), MAP_HF(HF/6-31G**), or IAP 
   - convolution process : important graph convolution
You can choose LAYER and TYPE in the following training and prediction.  

### Training
The trained data were stored in ./ckpt directory. 
If users want to train new data, please execute the following:

   ./training.sh TYPE LAYER

The model is trained using the training data listed in Opt1_acidic_tr.csv file.
You can set the atomic feature type (TYPE) and the number of convolution layers (LAYER).

### Prediction

There are two ways to predict pKa. 
If you have a target molecule, execute the following:

   ./prediction.sh CID group TYPE LAYER

The arguments, "CID" and "group", are listed in Opt1_acidic_tst.csv.

If you want to get the predicted pKa data about the molecules listed in Opt1_acidic_tst.csv, execute the following:

   ./loss_test.sh TYPE LAYER

The predicted data are stored in loss.log

#### (Example)
To predict the pKa value of "3-methylbenzenesulfonic acid" using MAP_HF feature with 5 convolution layers, 

   ./prediction.sh C617970 H19 MAP_HF 5

### Preparation of the concatenated vectors
The concatenated vectors can be prepared from sdf file.
The script file is stored in ./element_data/scripts directory.

#### (Example)
To prepare the concatenated vectors of the molecules in example.sdf,

   python mk_edata.py

The sdf file in example.sdf contains the following properties:

 1. atom type
 2. PKA_exp
 3. PKA_label
 4. wiberg bond index
 5. effective charge
 6. atomic polarizability
 7. atomic radius
 8. atomic ionization energy
 9. atimic electron affinity
 10. atom mass

The atom type is defined in AMBER parameter/topology file.
PKA_exp is the experimental data and PKA_label is the deprotonated hydrogen position.
The atomic data from 5 to 10 can be obtained from the following papers.
The wiberg bond index can be computed with Gaussian program coupled with NBO program.

At the current version, the script file can prepare IAP concatenated vectors.

### Analysis using Integrated Gradients (IGs)

If you want to perform the analysis using Integrated Gradients, execute the following:

   ./IG.sh

The obtained IGs are decomposed into the atomic properties at the k-th layer.

## Experimental data 
The experimental data in Opt1_acidic_tst.csv were taken from the previous study (DOI: [https://doi.org/10.1186/s13321-019-0384-1](https://doi.org/10.1186/s13321-019-0384-1)). 
When you use the experimental data, please cite their study. 

## Requirements

* python 3.7.13
* numpy 1.21.5
* pandas 1.3.5
* pytorch_lightning 1.7.7
* torch 1.13.1
* tqdm 4.64.1
