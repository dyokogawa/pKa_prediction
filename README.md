# Prediction of pKa from the concatenated vector
This is the code and data for the paper, "Feature selection in molecular graph neural networks based on quantum chemical approaches". This repository contains the code to predict pKa from the concatenated vector H<sup>(L)</sup>.
The concatenated vectors about the molecules listed in Opt1_acidic_tst.csv were stored in element_data.
We can obtain the predicted pKa about the molecules in Opt1_acidic_tst.csv.

## Instructions

To perform the prediction, execute the following:

   ./prediction.sh CID group

The arguments, "CID" and "group", are listed in Opt1_acidic_tst.csv.

### (Example)
To predict the pKa value of "3-methylbenzenesulfonic acid", 

   ./prediction.sh C617970 H19


## Details of the concatenated vectors
H<sup>(L)</sup> in element_data were prepared under the following condistions: 
   - number of convolution layers (L) : 10
   - atomic feature: MAP (the properties were prepared with CAM-B3LYP/aug-cc-pVDZ.)
   - convolution process : important graph convolution

## Experimental data 
The experimental data in Opt1_acidic_tst.csv were taken from the previous study (DOI: [https://doi.org/10.1186/s13321-019-0384-1](https://doi.org/10.1186/s13321-019-0384-1)). 
When you use the experimental data, please cite their study. 

## Limitations
At the current version, the script files to prepare the concatenated vectors. 
This is because the scrips employ the quantum chemical packages (Gaussian and GAMESS). 
