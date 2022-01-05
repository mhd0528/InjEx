#!/bin/bash
#SBATCH --job-name=ComplEx_NNE_AER_Test
#SBATCH --output=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/11.14_exps.out
#SBATCH --error=/blue/daisyw/ma.haodi/ComplEx-Inject/logs/11.14_exps.err
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

singularity exec --nv --bind ./ComplEx-Inject/:/home/ComplEx-Inject kbc_models_version1.8.sif python /home/ComplEx-Inject/kbc/process_datasets_cons.py
exit