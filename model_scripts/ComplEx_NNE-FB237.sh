#!/bin/bash
#SBATCH --job-name=ComplEx_NNE-FB237-all-Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/FB237/ComplEx_NNE-FB237-combine-all-1.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/FB237/ComplEx_NNE-FB237-combine-all.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks=1 # one process per node
#SBATCH --cpus-per-task=2 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10gb
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

echo "starting job"
echo "train: FB237; test: FB237; valid: FB237; rule: AnyBurl-entailment"
module load singularity

T1=`date +%Y %m %d-%H %M %S`
echo ${T1}
singularity exec --nv --bind /blue/daisyw/ma.haodi/ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset FB237 --model ComplEx_NNE --rank 50 --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 1e-2 --max_epochs 100 --valid 20 --rule_type 0 --mu 0.1
T2=`date +%Y %m %d-%H %M %S`

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
printf '%dd %dh:%dm:%ds\n' $(($ELAPSED/86400)) $(($ELAPSED%86400/3600)) $(($ELAPSED%3600/60)) $(($ELAPSED%60))