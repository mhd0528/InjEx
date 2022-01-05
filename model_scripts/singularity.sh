pip --version
nvidia-smi
singularity exec --bind ./ComplEx-Inject/:/home/ComplEx-Inject kbc_models_version1.8.sif python /home/ComplEx-Inject/kbc/process_datasets_cons.py