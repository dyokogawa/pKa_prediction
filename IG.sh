#!/bin/bash
#
# CLUSTERING: Kmeans or PIC
#

FNAME="C99967_H_16.dat"

DIR="./element_data/"
KBOND=10
NPROP=9
#
# REFERENCE FILE IN ${REF_DIR} 
#
REF_FNAME="C100027_H_15.dat"

if [ -e IG.tmp ]; then
  rm -f IG.tmp
  touch IG.tmp
fi

HYPER_LIST=("BEST1" "BEST2" "BEST3" "BEST4" "BEST5")
for HP in ${HYPER_LIST[@]}; do
   CKPT2="./ckpt/best_loss_${HP}.ckpt"
   CKPT3="./ckpt/maxval_${HP}.ckpt"
#
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
   pKa=`python IG_main.py ${DIR} ${REF_DIR} --kbond ${KBOND} --nprop ${NPROP} --ckpt2 ${CKPT2} --ckpt3 ${CKPT3} --fname ${FNAME} --nf0 ${NF0} --nf1 ${NF1} --ref_fname ${REF_FNAME}`
  echo "pKa" ${pKa}

   cat IG.csv >> IG.tmp
done

python IG_avg.py --kbond ${KBOND} --nprop ${NPROP} 

rm -f IG.tmp
rm -f IG.csv
