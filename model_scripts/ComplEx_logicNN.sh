#!/bin/bash
#SBATCH --job-name=ComplEx_NNE_logicNN_FB15K_Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_logicNN_grounding_type_0_0.8_rule_pi_dynamic_FB15K_test.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/ComplEx_logicNN_grounding_type_0_0.8_rule_pi_dynamic_FB15K_test.err
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
module load singularity

T1=`date +%Y%m%d-%H%M%S`
echo ${T1}
singularity exec --nv --bind ./ComplEx-Inject/:/home/ComplEx-Inject /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/ComplEx-Inject/kbc/learn_logicNN.py --dataset FB15K --model ComplEx_logicNN --rank 50 --optimizer Adagrad --learning_rate 0.1 --batch_size 2000 --regularizer F2 --reg 1e-3 --max_epochs 100 --valid 10 --rule_type 0
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
