#!/bin/bash

#SBATCH -o ./logs/launch_hparam_search_distance_healthy_cite_unif.out

#SBATCH -e ./logs/launch_hparam_search_distance_healthy_cite_unif.err

#SBATCH -J launch_hparam_search_distance_healthy_cite_unif

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

conda activate sc_exp_design 

cd /home/icb/alessandro.palma/environment/scportrait_ot/src

# python main.py --multirun datamodule=tonsilitis_cite hparam_search=sweep_distance logger.project=scportrait_ot_sweep_distance_tonsilitis_cite_non_unif
python main.py --multirun datamodule=tonsilitis_cite hparam_search=sweep_distance logger.project=scportrait_ot_sweep_distance_tonsilitis_cite datamodule.uniform_sampling_rna=True