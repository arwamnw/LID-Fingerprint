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

source /afs/cad.njit.edu/research/ccs/oria/2/Clustering/nnfdes/src/paths.sh

pwd=${p_datasets}Google-23/
ped=${p_data}Google-23/
dataset=faces
t=400
spa=5
rho=1.0
Iter=50

#vvvv
K=10
#^^^^


${franc} \
    -w ${pwd} -e ${ped} -n ${dataset}  \
    -K ${K} -S ${rho} \
    -I ${Iter} -z ${spa} -t ${t} \
