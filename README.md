# Testing for Fault Diversity in Reinforcement Learning

This repository contains all the material to replicate the results of the extended **AST 2024** paper: *Testing for Fault Diversity in Reinforcement Learning*.
Since this extension includes the reproduction of the initial experiments, two of the three use cases stem from the original evaluation of the [work of MDPFuzz](https://github.com/Qi-Pang/MDPFuzz).
However, compared to the AST, conference paper, in this work we do not use their implementation, but rather the replica released as a Python package [there](https://github.com/QuentinMaz/MDPFuzz_Replication).
Experiments were executed on a Linux machine (Ubuntu 22.04.5 LTS) equipped with an AMD EPYC 7302P 16-Core processor and 125GB of RAM.
Because of previous issues, we systematically the number of threads of the Pytorch library to 1.
If you are not aware of any related issue on your system, feel free to change this in the following scripts.
As a last note, in the following we provide instructions and scripts for each experiment, where each command runs a **single** execution.
While being cumbersome (some experiments involve hundreds of executions/commands), we think that it will let the users manage their executions according to their computation setups and infrastructure.

## Installation

Setup the environment for the Bipedal Walker and Lunar Lander use cases:
```bash
conda create -n exp_env python=3.6.3
conda env update --name exp_env --file python_environment.yml
conda activate exp_env
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```

Follow the instructions in `highway/README.md` to install the third use case.
*Remind to activate the correct environment and to navigate to the correct folder (`experiments/` or `highway/` when the scripts below!*

## Running the Experiments

In this extended work, there are a total of **four** sets of experiments to run, which respectively address:
1. RQ1&2 in the deterministic setting.
2. RQ3.
3. RQ1&2 in the stochastic setting.
4. RQ4.

The last set is by far the most time and resource demanding, as it involves dozens of 50K executions (per framework, use case and $(n,\text{ }number\_iterations)$ configuration)).


### RQ1&2: Deterministic Setting

First, navigate to the `experiments/` folder with `cd experiments`.
Then, run all the frameworks with Bipedal Walker and Lunar Lander:
```python
python ast_runner.py --use_case ll --method rt --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case ll --method ns --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case ll --method qd --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case ll --method mdpfuzz --seed_index 0 --descriptors 0 1

python ast_runner.py --use_case bw --method rt --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case bw --method ns --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case bw --method qd --seed_index 0 --descriptors 0 1
python ast_runner.py --use_case bw --method mdpfuzz --seed_index 0 --descriptors 0 1
```
Proceed this way for the nine remaining seed indices ($\in [1,...,9]$).

Then, navigate to the `highway/` folder with `cd ../highway` (and activate the Conda environment with `conda activate hw`).
There, follow a similar procedure. The arguments `use_case` and `descriptors` should however be removed:
```python
python ast_runner.py --method rt --seed_index 0
python ast_runner.py --method ns --seed_index 0
python ast_runner.py --method qd --seed_index 0
python ast_runner.py --method mdpfuzz --seed_index 0
# ... for the other seed indices
```

These calls use the default results folders (`results/`), which are assumed by the other scripts for analysing the raw data and plotting the analysis.

There are a total of $120$ commands ($3\text{ }use\_cases * 4\text{ }methods * 10\text{ }seeds$).

### RQ3: Bipedal Walker with different Expert Behaviors

This second set of experiments aims to assess the effect of the expert space definition.
This is done by running the QD-based frameworks (`ns` and `qd`) on Bipedal Walker with three different expert spaces.
The latter are passed through the `descriptors` arguments as feature indices ($\in {[2,\text{ }3], [5,\text{ }11], [4,\text{ }8]}$):
```python
python ast_runner.py --use_case bw --method ns --seed_index 0 --descriptors 2 3 --log_folder results/rq3
python ast_runner.py --use_case bw --method qd --seed_index 0 --descriptors 2 3 --log_folder results/rq3
# ... for the other seed indices

python ast_runner.py --use_case bw --method ns --seed_index 0 --descriptors 5 11 --log_folder results/rq3
python ast_runner.py --use_case bw --method qd --seed_index 0 --descriptors 5 11 --log_folder results/rq3
# ... for the other seed indices

python ast_runner.py --use_case bw --method ns --seed_index 0 --descriptors 4 8 --log_folder results/rq3
python ast_runner.py --use_case bw --method qd --seed_index 0 --descriptors 4 8 --log_folder results/rq3
# ... for the other seed indices
```
*As you can see, we indicate the results' folder (with `log_folder`) as `results/rq3`.*
There are a total of $60$ commands ($3\text{ }descriptor\_indices * 2\text{ }methods * 10\text{ }seeds$).

*NB: One can note that we do not re-execute Random Testing and MDPFuzz\*. Indeed, they do not depend on the behavior space (behaviors are just collected to report on the related metrics). The plotting script will take care of duplicating the frameworks' results.*

### RQ1&2: Stochastic Setting

This third set of experiments mimics the first one for the stochastic, which involves the following differences:
1. Test cases are now evaluated three times ($n=3$).
2. QD-based frameworks are tested with **four** implementations of our proposed *generic* (domain-agnostic) descriptors.

As such, not only the executions are roughly three times longer (the testing budget is also 5K iterations), but there are $4$ MAP-Elites (`qd`) and $4$ Novelty Search (`ns`) configurations to run for each use case and seed, hence increasing the total number of frameworks to run to $300$ ($3\text{ }use\_cases * 10\text{ }methods * 10\text{ }seeds$).

We provide another script for running the experiments of the stochastic setting — to ease user's experience —, even though the scripts rely on the same implementation.
As shown below, the results' folder for this set of experiments is `results_new/`.

#### Commands for Random Testing and MDPFuzz*
As previously mentioned, the behavior space does not impact these frameworks.
Therefore, instructions are for a given seed (here, $0$):
```python
# in `experiments/` and the Conda environment `exp_env`
python runner.py --use_case ll --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --use_case ll --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new

python runner.py --use_case bw --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --use_case bw --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
```
... And in a similar fashion, for Highway:
```python
# in `highway/` and the Conda environment `hw`
python runner.py --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
```

#### Commands for MAP-Elites and Novelty Search
Here are the instructions with the seed $0$ and descriptor space $[action\_entropy\_mean,\text{ }length\_spread]$:
```python
# in `experiments/` and the Conda environment `exp_env`
python runner.py --use_case ll --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --use_case ll --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new

python runner.py --use_case bw --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --use_case bw --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
```
The three other pairs of behavior descriptors to run (with the $10$ seeds) are $[action\_entropy\_mean,\text{ }length\_mean]$, $[action\_entropy\_argmax,\text{ }length\_mean]$ and $[action\_entropy\_argmax,\text{ }length\_spread]$.

For Highway, the $8$ instructions for a given seed (here, $0$) are:
```python
# in `highway/` and the Conda environment `hw`
python runner.py --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_mean length_mean --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_argmax length_mean --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_argmax length_spread --log_folder results_new

python runner.py --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_mean length_mean --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_argmax length_mean --log_folder results_new
python runner.py --method qd --seed_index 0 --descriptors action_entropy_argmax length_spread --log_folder results_new
```

### RQ4: Different $n$ values for a same testing budget

This last set of experiments aims to explore the relevance of accounting for the MDP's stochasticity.
In other words, this RQ explores the trade off between testing through diverse initial scenarios (i.e. test cases) and the MDP's dynamics.
To do so, the total testing budget is increased to 50K evaluations, which is spent for different pairs of $(n,\text{ }number\_iterations)$.
Precisely, we study **four** configurations: $\{(1,\text{ }50K),\text{ }(3,\text{ }16.67K),\text{ }(5,\text{ }10K),\text{ }(10,\text{ }5K)\}$.

In order to reduce the number of framework configuration to run — and given the robustness of QD-based testing against the implementation of the generic descriptors (see the paper) —, we use a single behavior space: $[action\_entropy\_mean,\text{ }length\_spread]$.
Therefore, there a total of $480$ framework executions ($3\text{ }use\_cases * 4\text{ }configurations * 4\text{ }methods * 10\text{ }seeds$).
In the following, we illustrate how to proceed the three use cases for a given seed ($0$) and configuration ($(1,\text{ }50K)$):
```python
# in `experiments/` and the Conda environment `exp_env`
python runner.py --use_case bw --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case bw --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case bw --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case bw --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000

python runner.py --use_case ll --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case ll --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case ll --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --use_case ll --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000

# in `highway/` (cd `../highway`) and the Conda environment `hw`
python runner.py --method qd --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --method ns --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --method rt --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
python runner.py --method mdpfuzz --seed_index 0 --descriptors action_entropy_mean length_spread --log_folder results_n_1 --env_seeds 1 --test_budget 50000
```
Repeat the commands above for each seed.
When proceeding to another configuration, don't forget to update the name of the results folder accordingly to $n$ (e.g. $results\_n\_3$).

## Computating the results analysis

**If you change the name of the results folders used above, you will have to update the scripts in the following procedure accordingly.**

### Deterministic Setting

Navigate to the `experiments/` folder with `cd experiments` and activate the `exp_env` environment.
To perform the analysis of the results and plot the subsequent figures, simply run `python ast_result_analysis.py`.
The script will store the results of RQ1&2 under `data_ast/`, with the figure of the paper at `data_ast/rq.png`.
For RQ3, the results are saved under `data_ast_rq3/`, and the figure will be located at `data_ast_rq3/rq3.png`.

### Stochastic Setting

In a similar way as above, the computation of the results analysis for RQ1&2 can be done with `python result_analysis.py`.
The results will be stored at `data_new/` and the figure at `data_new/rq.png`.

As for the last research question RQ4, first compute the results for each configuration:
```python
python compute_study.py --data_folder results_n_1 --result_folder data_n_1
python compute_study.py --data_folder results_n_3 --result_folder data_n_3
python compute_study.py --data_folder results_n_5 --result_folder data_n_5
python compute_study.py --data_folder results_n_10 --result_folder data_n_10
```
Then, assemble the results and plot the boxplots used in the paper with `python plot_rq4.py`.
The figures will be saved to current directory (`experiments/`) at `boxplot_fault.png` and `boxplot_coverage.png`.