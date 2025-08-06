"""
Script to study the effect of rolling the episode until termination, instead of the current 300 time step limit.
I suspect a much narrower behavior space, as the descriptors are averages of observation features...
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from common import load_bipedal_walker_model
import gym



def execute_policy(input, model, env_seed, deterministic=True, timelimit: int = 1000):
        env = gym.make("BipedalWalkerHardcore-v4", rand_seed=env_seed)

        acc_reward = 0.0
        features = np.zeros(12)
        current_features = np.zeros(12)

        obs = env.reset(input)
        state = None

        for t in range(timelimit):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)


            features += info["features"]  # numpy array
            acc_reward += reward

            if done:
                break

            if t < 300:
                current_features += info["features"]  # numpy array

        env.close()

        features /= t
        # currently, we iterate until 300
        current_features /= min(t, 299)

        return (
            acc_reward,
            (reward == -100),
            features,
            current_features,
            t + 1 # length of the episode
        )


if __name__ == "__main__":
    torch.set_num_threads(1)

    # parameters
    env_seed = 0
    num_inputs = 10_000
    input_size = 10
    rng = np.random.default_rng(0)  # type: np.random.Generator
    inputs = rng.integers(1, 4, size=(num_inputs, input_size))

    model = load_bipedal_walker_model()

    log_folder_name = "bw_behaviors"
    log_folder = Path(log_folder_name)
    log_folder.mkdir(parents=True, exist_ok=True)


    features_buffer = open(log_folder / "features.txt", "w", buffering=1)
    current_features_buffer = open(log_folder / "current_features.txt", "w", buffering=1)
    inputs_buffer = open(log_folder / "inputs.txt", "w", buffering=1)
    logs_buffer = open(log_folder / "logs.csv", "w", buffering=1)
    # header
    print("acc_reward,failed,ep_length", file=logs_buffer)

    for input in inputs:
        np.savetxt(inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")

        acc_reward, failed, features, current_features, ep_length = execute_policy(input, model, env_seed, deterministic=True, timelimit=1000)

        for arr, buffer in zip([features, current_features], [features_buffer, current_features_buffer]):
            np.savetxt(buffer, arr.reshape(1, -1), delimiter=",")

        print(f"{acc_reward},{int(failed)},{ep_length}", file=logs_buffer)

    for buffer in [features_buffer, current_features_buffer, inputs_buffer, logs_buffer]:
        buffer.close()