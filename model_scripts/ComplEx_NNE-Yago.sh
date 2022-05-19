#!/bin/bash
#SBATCH --job-name=ComplEx_NNE-Yago-all-Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/Yago/ComplEx_NNE-Yago-combine-all_1.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/Yago/ComplEx_NNE-Yago-combine-all.err
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
# echo "train: FB237; test: FB237; valid: FB237; rule: AnyBurl-entailment"
module load singularity

T1=`date +%Y%m%d-%H%M%S`
echo ${T1}
singularity exec --nv --bind /blue/daisyw/ma.haodi/ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset YAGO3-10 --model ComplEx_NNE --rank 50 --optimizer Adagrad --learning_rate 2e-1 --batch_size 1000 --regularizer N3 --reg 1e-3 --max_epochs 100 --valid 20 --rule_type 0 --mu 0.1
T2=`date +%Y%m%d-%H%M%S`

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
echo $(date -ud "@$ELAPSED" +'$((%s/3600/24)) days %H hours %M minutes %S seconds')