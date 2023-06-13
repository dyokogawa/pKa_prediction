#!/bin/bash
#
# CLUSTERING: Kmeans or PIC
#

FEATURE_TYPE=$1
KBOND=$2

ELIST="H"
REF_LIST="Opt1_acidic_tr.csv"

DIR="./element_data/${FEATURE_TYPE}/${KBOND}/"
NEPOCH=3000
if [ ${FEATURE_TYPE} = "MAP_CAM" -o ${FEATURE_TYPE} = "MAP_HF" ]; then
   NPROP=9
else
   NPROP=6
fi
#
# HYPER PARAMETERS
#
NF_LIST="./hp_parameters/${FEATURE_TYPE}/${KBOND}/NF_LIST"

NF0=`grep BEST1 ${NF_LIST} | awk '{print $2}'`
NF1=`grep BEST1 ${NF_LIST} | awk '{print $3}'`
RATIO=`grep BEST1 ${NF_LIST} | awk '{print $4}'`
#
# 
#
python training.py ${DIR} --elist ${ELIST} --kbond ${KBOND} --nprop ${NPROP} --epochs ${NEPOCH} --ref_list ${REF_LIST} --nf0 ${NF0} --nf1 ${NF1} --p ${RATIO}
