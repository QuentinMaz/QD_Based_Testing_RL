# Highway Use Case

## Installation

First, setup the following Python environment with [Conda](https://anaconda.org/anaconda/conda):
```bash
conda create -n hw python=3.10.12
conda activate hw
pip install -r highway_requirements.txt
```

Then, install the hardware-specific [Pytorch](https://pytorch.org/) packages:
```bash
pip install torch torchvision torchaudio
```

## Experiments

To perform all the sets of experiments of the paper, please refer to `../README.md`.