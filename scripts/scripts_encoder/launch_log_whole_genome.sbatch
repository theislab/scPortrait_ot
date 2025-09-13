#!/bin/bash

#SBATCH -o ./logs/decoder_log_whole_genome.out

#SBATCH -e ./logs/decoder_log_whole_genome.err

#SBATCH -J decoder_log_whole_genome 

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH --mem=196G

#SBATCH --nice=0

#SBATCH --time=1-00:00

export HYDRA_FULL_ERROR=1

cd /home/icb/alessandro.palma/environment/scportrait_ot/src

python main_autoencoder_train.py \
  adata_path="/home/icb/alessandro.palma/environment/scportrait_ot/project_folder/data/citeseq/CITEseq_rna_processed_full_genome.h5ad" \
  count_label="X_normalized_log" \
  dims="[256, 256]" \
  project="decoder_cite_log_whole_genome" \
  likelihood="gaussian"
