#!/bin/bash
#
#  FNAME: file name in element_data (CID_H_{H position}.dat)
#
CID=$1
group=$2
#
# set FNAME
#
num=${group:1}
FNAME=${CID}"_H_"${num}.dat
#
#
#
REF_LIST="Opt1_acidic_tst.csv"
DIR="./element_data/"
KBOND=10
NPROP=9
HIDDEN_DIM=20
#
#
#
if [ -e pKa.tmp ]; then
  rm -f pKa.tmp
  touch pKa.tmp
fi

HYPER_LIST=("BEST1" "BEST2" "BEST3" "BEST4" "BEST5")
for HP in ${HYPER_LIST[@]}; do
   CKPT2="./ckpt/best_loss_${HP}.ckpt"
   MAXVAL="./ckpt/maxval_${HP}.ckpt"

   if [ ${HP} = "BEST1" ]; then
      NF0=140
      NF1=170
   elif [ ${HP} = "BEST2" ]; then
      NF0=150
      NF1=130
   elif [ ${HP} = "BEST3" ]; then
      NF0=140
      NF1=100
   elif [ ${HP} = "BEST4" ]; then
      NF0=140
      NF1=160
   elif [ ${HP} = "BEST5" ]; then
      NF0=120
      NF1=130
   fi
#
#  
#
   python prediction.py ${DIR} --kbond ${KBOND} --nprop ${NPROP} --hidden_dim ${HIDDEN_DIM} --ckpt2 ${CKPT2} --ckpt3 $MAXVAL --fname ${FNAME} --ref_list ${REF_LIST} --nf0 ${NF0} --nf1 ${NF1} >> pKa.tmp

done

#
#  ensemble average
#
pred_pKa=`cat pKa.tmp | awk '{sum+=$1} END {print sum/NR}'`
ref_pKa=`cat pKa.tmp | awk '{sum+=$2} END {print sum/NR}'`
#
#  results
#
printf 'predicted pKa: %.3f, reference pKa: %.3f\n' ${pred_pKa} ${ref_pKa}
#
#  remove temporary file
#
rm -f pKa.tmp
