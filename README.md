# pKa_prediction
This is the code and data for the paper, "Feature selection in molecular graph neural networks based on quantum chemical approaches". This repository contains the code to predict pKa from the concatenated vector H^(L).
The concatenated vectors about the molecules listed in Opt1_acidic_tst.csv were stored in element_data.
We can obtain the predicted pKa about the molecules in Opt1_acidic_tst.csv.

Instructions

execute the following:

   ./prediction.sh CID group

"CID" and "group" are listed in Opt1_acidic_tst.csv.

Example:
   ./prediction.sh C617970 H19


Details of the concatenated vectors

   number of convolution layers (L) : 10
   atomic feature: MAP (the properties were prepared with CAM-B3LYP/aug-cc-pVDZ.)
   convolution process : important graph convolution
