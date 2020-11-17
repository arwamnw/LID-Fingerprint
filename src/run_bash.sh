#$ -M amw7@njit.edu
#$ -N TestBatch
#$ -m abe
#$ -cwd
#$ -j y
#$ -r y
#$ -S /bin/bash
source=/phenome/amw7/NNFDecsent_Wighted_ID_fix_r/src/CIFAR-100.txt
#mpirun -np $NSLOTS /path/to/your/mpi/prog
bash bash.sh >> $source