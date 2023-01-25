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
