#!/bin/bash

#$ -M js87@njit.edu
#$ -m be
#$ -cwd
#$ -j y
#$ -r y
#$ -S /bin/bash

#vvvv <k>
#$ -N exp1107all-tp-k10-g23-franc-0y-t400
#^^^^

#source /afs/cad.njit.edu/research/ccs/oria/2/Clustering/nnfdes/src/bash.sh
franc=/LID-Fingerprint/src/franc
p_datasets=/LID-Fingerprint/
p_data=/LID-Fingerprint/
pwd=${p_datasets}ALOI-100-Dataset/
ped=${p_data}ALOI-100-Dataset/
dataset=ALOI-100
t=400
spa=5
rho=1.0
Iter=70

z=0.0025
echo ${dataset}
spa=$((python /LID-Fingerprint/Dataset_Editing/calculate_z.py ${pwd} ${dataset} ${z}) 2>&1)
echo ${spa}
K=100
echo ${K}

#echo factor 8
#^^^^

${franc} \
    -w ${pwd} -e ${ped} -n ${dataset} \
    -K ${K} -S ${rho} \
    -I ${Iter} -z ${spa} -t ${t} \
