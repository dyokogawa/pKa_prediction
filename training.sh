#!/bin/bash
#
# CLUSTERING: Kmeans or PIC
#

KBOND=10
TYPE="M"
ELIST="H"
REF_LIST="Opt1_acidic_tr.csv"

NPROP=9
DIR="./element_data/"
NEPOCH=3000
#
# HYPER PARAMETERS
#
NF0=140
NF1=170
RATIO=0.14159586227379684
#
# 
#
python training.py ${DIR} --elist ${ELIST} --kbond ${KBOND} --nprop ${NPROP} --epochs ${NEPOCH} --ref_list ${REF_LIST} --nf0 ${NF0} --nf1 ${NF1} --p ${RATIO}
