#!/bin/bash

FEATURE_TYPE=$1
KBOND=$2

REF_LIST="Opt1_acidic_tst.csv"
DIR="./element_data/${FEATURE_TYPE}/${KBOND}/"
HIDDEN_DIM=20
if [ ${FEATURE_TYPE} = "MAP_CAM" -o ${FEATURE_TYPE} = "MAP_HF" ]; then
   NPROP=9
else
   NPROP=6
fi

if [ -e loss.log ]; then
  rm -f loss.log
fi

HYPER_LIST=("BEST1" "BEST2" "BEST3" "BEST4" "BEST5")

while read line
do
  mol=`echo $line|awk -F "," '{print $1}'`
  if [ $mol != "CID" ]; then
    label=`echo $line|gawk -v FPAT='([^,]+)|(\"[^\"]+\")' '{print $(NF-1)}'`
    site=`echo $label|sed -e 's/[^0-9]//g'`
    atom=`echo $label|sed -e 's/[0-9]//g'`

    FNAME="${mol}_${atom}_${site}.dat"

    if [ -e pKa.tmp ]; then
      rm -f pKa.tmp
      touch pKa.tmp
    fi

    for HP in ${HYPER_LIST[@]}; do
      CKPT2="./ckpt/${FEATURE_TYPE}/${KBOND}/best_loss_${HP}.ckpt"
      MAXVAL="./ckpt/${FEATURE_TYPE}/${KBOND}/maxval_${HP}.ckpt"
      NF_LIST="./hp_parameters/${FEATURE_TYPE}/${KBOND}/NF_LIST"

      NF0=`grep ${HP} ${NF_LIST} | awk '{print $2}'`
      NF1=`grep ${HP} ${NF_LIST} | awk '{print $3}'`
#
#  
#
      python prediction.py ${DIR} --kbond ${KBOND} --nprop ${NPROP} --hidden_dim ${HIDDEN_DIM} --ckpt2 ${CKPT2} --ckpt3 $MAXVAL --fname ${FNAME} --ref_list ${REF_LIST} --nf0 ${NF0} --nf1 ${NF1} >> pKa.tmp
    done
#
#   ensemble average
#
    pred_pKa=`cat pKa.tmp | awk '{sum+=$1} END {print sum/NR}'`
    ref_pKa=`cat pKa.tmp | awk '{sum+=$2} END {print sum/NR}'`
#
#   results
#
    printf 'predicted pKa: %.3f, reference pKa: %.3f\n' ${pred_pKa} ${ref_pKa} >> loss.log
#
#  remove temporary file
#
    rm -f pKa.tmp

  fi
done < $REF_LIST
