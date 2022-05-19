#!/bin/bash
#SBATCH --job-name=ComplEx-FB237-1_shot-Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/Yago/ComplEx-Yago-all-2.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/Yago/ComplEx-Yago-all.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks=1 # one process per node
#SBATCH --cpus-per-task=4 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10gb
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

echo "starting job"
# echo "train: FB237-0404-1; test: FB237-0404-1; valid: FB237-0404-1; "
module load singularity

T1=`date +%Y%m%d-%H%M%S`
echo ${T1}
singularity exec --nv --bind /blue/daisyw/ma.haodi/ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset YAGO3-10 --model ComplEx --rank 50 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer F2 --reg 5e-3 --max_epochs 100 --valid 20
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"