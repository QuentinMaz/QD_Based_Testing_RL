import json
import sys
import os
import time
import torch
import tqdm
import pandas as pd
import numpy as np

from metrics import compute_action_distributions, compute_action_std, compute_entropy
from typing import List, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO
import gym


'''
Lunar Lander problem use case study.
Here, the input space is 2d, and it only describes the initial force applied to the lander.
'''

############################ CONSTANTS ################################


# default values
DEFAULT_MIN = -1000
DEFAULT_MAX = 1000
DEFAULT_MIN_INPUT = np.array([DEFAULT_MIN, DEFAULT_MIN])
DEFAULT_MAX_INPUT = np.array([DEFAULT_MAX, DEFAULT_MIN])
DEFAULT_MAX_DIST_INPUT: np.ndarray = np.linalg.norm(DEFAULT_MAX_INPUT - DEFAULT_MIN_INPUT)
MAX_TIME = 300


###################### EXECUTION/EXPERIMENT SUPPORTERS ################################


def generate_input(rng: np.random.Generator, lows: List[float], highs: List[float]):
    '''Generates a single input between the given bounds (parameters).'''
    return rng.uniform(low=lows, high=highs, size=2)


def generate_inputs(rng: np.random.Generator, lows: List[float], highs: List[float], n: int):
    '''Generates @n inputs with the lower and upper bounds parameters.'''
    return rng.uniform(low=lows, high=highs, size=(n, 2))


def load_lunar_lander_model():
    '''Loads the model under test.'''
    custom_objects = {
        'learning_rate': 0.0,
        'lr_schedule': lambda _: 0.0,
        'clip_range': lambda _: 0.0,
    }
    return PPO.load('rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip', custom_objects=custom_objects, device="cpu")


# can be used to study the case where we want to analyze the failures in details (i.e., the test input does include the seed + one execution)
def execute_policy(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 1000, deterministic: bool = True) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''Executes the model on the environment and only computes the hand-coded behavior. It also returns the final state.'''
    t0 = time.time()
    env: gym.Env = gym.make('LunarLander-v3')
    env.seed(env_seed)
    obs = env.reset(input)
    state = None
    acc_reward = 0.0

    actions = []

    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []

    for _ in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        acc_reward += reward

        actions.append(action)
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if done:
            break


    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)
    behavior = np.array([impact_x_pos, impact_y_vel])
    env.close()
    exec_time = time.time() - t0
    return acc_reward, (reward == -100), behavior, obs, exec_time, actions


# to study the case of testing overall agent's robustness (i.e., test input does not include the seed)
def execute_stochastic_policy(
        input: np.ndarray,
        model: BaseAlgorithm,
        env_seed: int,
        n: int,
        sim_steps: int = 1000
        ) -> Tuple[float, float, np.ndarray, np.ndarray, dict]:
    '''Executes n times a stochastic model and returns the results for each metric as lists.'''
    rewards, failures, actions = [], [], []
    # additional metrics
    final_obs_list, behavior_list = [], []
    for _ in range(n):
        acc_reward, failed, behavior, final_obs, exec_time, action_seq = execute_policy(input, model, env_seed, deterministic=False, sim_steps=sim_steps)
        rewards.append(acc_reward)
        failures.append(failed)
        actions.append(action_seq)
        behavior_list.append(behavior)
        final_obs_list.append(final_obs)

    # metrics for possible generic behavior space
    ep_length = [len(l) for l in actions]
    action_dist = compute_action_distributions(actions, range=[0, 4], bins=4)

    measures = dict(
        length_mean = np.mean(ep_length),
        length_std = np.std(ep_length),
        length_spread = max(ep_length) - min(ep_length),
        action_std = compute_action_std(actions),
        action_entropy = compute_entropy(action_dist)
    )

    return np.mean(rewards), np.mean(failures), np.vstack(behavior_list), np.vstack(final_obs_list), measures


def execute_policy_trajectory(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 1000) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''
    Executes the model with the simulator and returns state sequence. Useful for MDPFuzz.
    '''
    t0 = time.time()
    env: gym.Env = gym.make('LunarLander-v3')
    env.seed(env_seed)
    obs = env.reset(input)
    state = None
    acc_reward = 0.0

    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []

    obs_seq = []

    for _ in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        acc_reward += reward

        obs_seq.append(obs)

        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if done:
            break


    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)
    behavior = np.array([impact_x_pos, impact_y_vel])
    env.close()
    exec_time = time.time() - t0

    return acc_reward, (reward == -100), behavior, np.array(obs_seq), exec_time


def get_edges(env_seed: int, sim_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(f'grid/ll/{env_seed}_{sim_steps}_xedges.npy'), np.load(f'grid/ll/{env_seed}_{sim_steps}_yedges.npy')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_lunar_lander_model()

    rng: np.random.Generator = np.random.default_rng(main_seed)

    oracles, rewards, behaviors, final_states = [], [], [], []
    inputs = []
    for _ in tqdm.tqdm(range(100)):
        input: np.ndarray = rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=2)
        r, o, b, fs, _ = execute_policy(input, model, env_seed, 1000)
        oracles.append(o)
        rewards.append(r)
        behaviors.append(b)
        final_states.append(fs)

        inputs.append(input)

    np.save('behaviors.npy',np.vstack(behaviors))
    np.save('inputs.npy', np.vstack(inputs))
    import matplotlib.pyplot as plt
    print('test OK')
    fig, ax = plt.subplots()
    behaviors = np.array(behaviors)
    ax.scatter(behaviors[:, 0], behaviors[:, 1], s=10, alpha=0.5)
    fig.savefig('ll_test.png')