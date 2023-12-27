<div align="center">

# Multi-label classification for movie genres

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

We conducted the classification of movie genres on a dataset extracted from the MovieLens 10M dataset, using (macro) precision, recall, and F1-score to evaluate performance. You can have more information at [video presentation](https://youtu.be/oO9o4on_0Gw)

## Installation

```bash
# clone project
git clone https://github.com/PAD2003/PAD2003-multi-label-movie-genres.git
cd your-repo-name

# create conda environment and install dependencies
conda create --name mlmg python=3.8.17
conda activate mlmg
pip install -r requirements
```

## How to train

Look into configs/experiment to choose the config you want to use for your training process.

```bash
python src/train.py experiment=<experiment_name>.yaml
```

## How to validation

Look into validation.ipynb. You only need to adjust config and then run cells.

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
