# scPortrait OT

This code implements the translation from CODEX features to single-cell gene expression. The repository contains the code for the OT-based Conditional Flow Matching algorithm, which facilitates feature translation and enables the decoder to transition from HARMONY embeddings to gene expression.

# Install the software 

Clone the repository:

```
git clone https://github.com/theislab/scPortrait_ot.git
```

Install the environment with conda:

```
conda env create -f environment.yml
```

Activate environment 

```
conda activate scportrait_ot
```

Install environment 

```
cd directory_where_you_have_your_git_repos/scPortrait_ot
pip install -e . 
```

Create an experiment folder and download the main data from `scPortrait_manuscript`.

# Navigate the folder

* Model training is performed through a mix of [hydra](https://github.com/facebookresearch/hydra) (configurations and sweeps), [wandb](https://wandb.ai/site/) (logging) and PyTorch Lightning (model implementation and callbacks).   

* Model implementation can be found in `scPortrait_ot`. The configurations for training the reconstructing decoder and Flow Matching model are in `configs_autoencoder` and `configs`. The training processes are decoupled and can be run independently.

* To train models, make sure to complete all the configuration files with the paths to the data and target folder according to your file hierarchy. The files requiring path changes are:

  * `configs/datamodule/tonsilitis_cite.yaml`
  * `configs/training_config/default.yaml`
  * `configs/configs_autoencoder/train_cite_log.yaml`

* The model training scripts are in `scripts`; they automatically run a hyperparameter search on slurm. There are two subfolders, one for training the Flow Matching model and one for the decoder model. Make sure to create a sub-folder called `logs` inside these folders if running the scripts on slurm.
