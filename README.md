# GPS++: An Optimised Hybrid GNN/Transformer for Molecular Property Prediction

An optimised hybrid GNN/Transformer model for molecular property prediction using Graphcore IPUs, trained on the [PCQM4Mv2](https://arxiv.org/abs/2103.09430) dataset. The flexible hybrid model closely follows the [General, Powerful, Scalable (GPS) framework](https://arxiv.org/abs/2205.12454) and combines the benefits of both message passing and attention layers for graph-structured input data training.


### Dataset

The [PCQM4Mv2](https://arxiv.org/abs/2103.09430) dataset is a recently published dataset for the OGB Large Scale Challenge built to aide the development of state-of-the-art machine learning models for molecular property prediction. The task is for the quantum chemistry task of predicting the [HOMO-LUMO energy gap](https://en.wikipedia.org/wiki/HOMO_and_LUMO) of a molecule.

The dataset consists of 3.7 million molecules defined by their SMILES strings which can simply be represented as a graph with nodes and edges.

The dataset includes four splits: train, valid, test-challenge and test-dev. The train and valid splits have true label and can be used for model development. The test-challenge split is used for the OGB-LSC PCQM4Mv2 challenge submission and test-dev for the [leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) submission.

At the start of training, the dataset will be downloaded and the additional features will be preprocessed automatically, including the 3D molecular features that are provided with the dataset.

### Training and inference on OGB-LSC PCQM4Mv2

In order to begin training, select the configuration you wish to run from those in the `configs` directory.

Then run with the following command:

```shell
python3 run_training.py --config configs/<CONFIG_FILE>
```

After training has finished inference will follow and show the validation results on the validation dataset.

To run inference separately and on other dataset splits, use the following command, changing the `--inference_fold` flag:

```shell
python3 inference.py --config configs/<CONFIG_FILE> --checkpoint_path <CKPT_PATH> --inference_fold <DATA_SPLIT_NAME>
```

## Logging and visualisation in Weights & Biases

This project supports Weights & Biases, a platform to keep track of machine learning experiments. To enable this, use the `--wandb` flag.

The user will need to manually log in (see the quickstart guide [here](https://docs.wandb.ai/quickstart)) and configure these additional arguments:

- `--wandb_entity`
- `--wandb_projecy`

For more information please see https://www.wandb.com/.

## Licensing

This application is licensed under the MIT license, see the LICENSE file at the base of the repository.

This directory includes derived work, these are documented in the NOTICE file at the base of the repository.

The checkpoints are licensed under the Creative Commons CC BY 4.0 license.
