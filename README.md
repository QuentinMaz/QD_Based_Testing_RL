# Testing for Fault Diversity in Reinforcement Learning

This repository contains all the material to replicate the results of the paper.
The virtual environment comes from the [work of MDPFuzz](https://github.com/Qi-Pang/MDPFuzz).
Similarly, we use some parts of their implementation to replicate their methodology.
Experiments were executed on a Linux machine (Ubuntu 22.04.3 LTS) equipped with an AMD Ryzen 9 3950X 16-Core processor and 32GB of RAM.
Because of previous issues, we systematically the number of threads of the Pytorch library to 1.
If you are not aware of any related issue on your system, feel free to change this in the following scripts.
As a last note:
- The scripts run the experiments sequentially (for ease of replication and simplicity).
- However, the implementation of the multivariate gaussian models used by MDPFuzz is multithreaded (Scipy 1.5.4).

## Installation

Setup the environment (as indicated by the original repository):
```bash
conda create -n exp_env python=3.6.3
conda env update --name exp_env --file python_environment.yml
conda activate exp_env
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```

## Experiments

First, naviguate to the `experiments/` folder with `cd experiments`.

### Bipedal Walker

```python
python bw_framework.py
python bw_mdpfuzz.py
```
The first command runs the two QD-based policy testing implementations and the Random Testing baseline (the second one, MDPFuzz).

### Lunar Lander

```python
python ll_framework.py
python ll_mdpfuzz.py
```

### Taxi

```python
python tt_framework.py
python tt_mdpfuzz.py
```

The execution results are saved in `results/`.

## Mining the results

After the execution, the computation of the raw data and their presentation (e.g., charts) can be launched with the command `python result_analysis.py`.
The data of the figures are stored under the folder `data/` (such as the behavior coverage) and the previous command directly outputs the .png files of the figures in the current folder (`experiments/`).
Thanks for reading and do not hesitate to raise any replication problem to the maintainer of this repository.