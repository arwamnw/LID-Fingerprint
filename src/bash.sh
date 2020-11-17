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
franc=/home/a/amw7/Hill_ID_NNWID_Descent/NNFDecsent_Wighted_HillID_fix_r_neighbors_of_each_features_Print_Binary_data/src/franc
p_datasets=/home/a/amw7/
p_data=/home/a/amw7/
pwd=${p_datasets}RLCT/
ped=${p_data}RLCT/
dataset=RLCT
t=400
spa=5
rho=1.0
Iter=70

z=0.0025
echo ${dataset}
spa=$((/afs/cad/linux/anaconda-2.1.0/anaconda/bin/python /home/a/amw7/Sparsification-Kmeans/src/Dataset_Editing/calculate_z.py ${pwd} ${dataset} ${z}) 2>&1)
echo ${spa}
K=100
echo ${K}

#echo factor 8
#^^^^

${franc} \
    -w ${pwd} -e ${ped} -n ${dataset} \
    -K ${K} -S ${rho} \
    -I ${Iter} -z ${spa} -t ${t} \