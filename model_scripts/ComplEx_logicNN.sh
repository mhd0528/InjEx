#!/bin/bash
#SBATCH --job-name=ComplEx_NNE_logicNN_FB15K_Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_logicNN-type_4-FB237-0_shot.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_logicNN-type_4-FB237-0_shot.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks=1 # one process per node
#SBATCH --cpus-per-task=4 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100gb
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

echo "starting job"
echo "train: eval_1207; test: eval_1207; valid: eval_1207; rule: _entailment_cons-rq"
module load singularity

T1=`date +%Y%m%d-%H%M%S`
echo ${T1}
singularity exec --nv --bind ./ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset FB237 --model ComplEx_logicNN --rank 50 --optimizer Adagrad --learning_rate 0.1 --batch_size 2000 --regularizer F2 --reg 1e-3 --max_epochs 100 --valid 20 --rule_type 4
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
