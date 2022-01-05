#!/bin/bash
#SBATCH --job-name=ComplEx_NNE_logicNN_type_4_Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_FB15k_test.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_FB15k_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks=1 # one process per node
#SBATCH --cpus-per-task=8 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --mem-per-cpu=500M # cannot be '3.5G'. sbatch requires this parameter to be an integral number.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

echo "starting job"
module load singularity

T1=$(date +%s)
echo T1
singularity exec --nv --bind ./ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset FB15K --model ComplEx --rank 50 --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 --regularizer F2 --reg 1e-3 --max_epochs 100 --valid 20 --rule_type 4
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
