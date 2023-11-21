# Testing for Fault Diversity in Reinforcement Learning

This repository contains all the material to replicate the results. The whole repository builds upon the [MDPFuzz work](https://github.com/Qi-Pang/MDPFuzz). Indeed, this work explores the use of QD optimization for finding diverse faults in policies in which we compare MDPFuzz and Random Testing.

## Installation

Setup the environment (as indicated by the original repository):
```bash
conda create -n my_env python=3.6.3
conda env update --name my_env --file python_environment.yml
conda activate my_env
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```

## Experiments

### Running the experiments

TODO.


### Mining the results

TODO.