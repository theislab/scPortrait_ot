#!/bin/bash

# Parameters
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=6
#SBATCH --error=/ictstr01/home/icb/alessandro.palma/environment/scportrait_ot/src/multirun/2025-08-04/20-51-24/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=main
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/ictstr01/home/icb/alessandro.palma/environment/scportrait_ot/src/multirun/2025-08-04/20-51-24/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /ictstr01/home/icb/alessandro.palma/environment/scportrait_ot/src/multirun/2025-08-04/20-51-24/.submitit/%A_%a/%A_%a_%t_log.out --error /ictstr01/home/icb/alessandro.palma/environment/scportrait_ot/src/multirun/2025-08-04/20-51-24/.submitit/%A_%a/%A_%a_%t_log.err /home/icb/alessandro.palma/miniconda3/envs/sc_exp_design/bin/python -u -m submitit.core._submit /ictstr01/home/icb/alessandro.palma/environment/scportrait_ot/src/multirun/2025-08-04/20-51-24/.submitit/%j
