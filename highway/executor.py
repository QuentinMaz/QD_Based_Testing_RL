from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import os
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from device import DEVICE
from my_highway import (
    InitializableHighwayEnv,
    generate_input,
    generate_even_input,
    mutate_positions,
)
from metrics import (
    compute_action_distributions,
    compute_action_std,
    compute_entropy,
)
from agents import load_model

# directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_agent_path = "saved_models/dqn/rl_model_1900152_steps.zip"
# combines where the current file is with the relative path
TRAINED_AGENT_PATH = os.path.join(script_dir, relative_agent_path)


class TestManager(ABC):
    """
    API for managing test input generation, mutation and test case execution.
    Note that a TestManager also holds the agent (and might have a private environment too).
    """

    def __init__(self, num_max_steps: int, seed: int = None) -> None:
        self.num_max_steps = num_max_steps
        self.seed = seed
        super().__init__()

    @abstractmethod
    def mutate_input(
        self, input: np.ndarray, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        pass

    @abstractmethod
    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        pass

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.array([self.generate_input(rng) for _ in range(n)])

    @abstractmethod
    def execute_policy(
        self, input: np.ndarray, policy: Any
    ) -> Tuple[float, float, int, np.ndarray]:
        """
        Runs the policy with the environment initialized w.r.t `input`.

        Parameters
        ----------
        inputs : np.ndarray
            Data for setting a test case.
        policy : Any
            The agent under test. Provide None if the class already holds the latter.

        Returns
        -------
        failure probability : float
        accumulated reward : float
        episode length : int
        """
        pass

    @abstractmethod
    def load_policy(self, *args, **kwargs):
        pass


CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,  # 15 for the other models!
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
    },
    "collision_reward": -10.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 1.0,
    "lane_change_reward": 0.0,
    "reward_speed_range": [23, 30],
    "normalize_reward": False,
}


class HighwayTestManager(TestManager):

    def __init__(
        self,
        num_max_steps: int = 800,
        seeds: int = [None],
    ) -> None:
        super().__init__(num_max_steps, None)
        self.env = InitializableHighwayEnv(config=CONFIG, render_mode="rgb_array")
        self.seeds = seeds

    # def load_policy(self, *args, **kwargs):
    #     return DQN.load(TRAINED_AGENT_PATH, device=DEVICE)


    def load_policy(self, model_path: str = None, **kwargs):
        """
        Parameters
        ----------
        model_path : str
            Path to the model.

        Returns
        -------
        policy : Union[DQN, AgentWrapper]
        """
        if model_path is None:
            return DQN.load(TRAINED_AGENT_PATH, device=DEVICE)
        else:
            return load_model(
                model_path,
                model_type="DQNAgent" if "dqnagent" in model_path else "DQN"
            )



    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        # num_vehicles = kwargs.get("num_vehicles", 20)
        # num_lanes = kwargs.get("num_lanes", 3)
        num_vehicles = 20
        num_lanes = 3
        distance_between_vehicles = 25

        return generate_even_input(
            rng, num_vehicles, num_lanes, distance_between_vehicles
        )

    def mutate_input(
        self, input: np.ndarray, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        initial_distance = 25
        return mutate_positions(input, rng, initial_distance=initial_distance)

    def execute_policy(
        self,
        input: np.ndarray,
        policy: Any,
        deterministic: bool = True,
        seed: int = None,
    ) -> Tuple[float, float, int, np.ndarray]:
        failure, final_obs, action_seq, reward_seq, measures, _frames = (
            self._record_execution(
                policy, input, record=False, deterministic=deterministic, seed=seed
            )
        )
        return float(failure), np.sum(reward_seq), len(reward_seq)

    def execute_stochastic_policy(
        self, input: np.ndarray, policy: Any, n: int, deterministic: bool = False
    ) -> Tuple[float, float, list[np.ndarray], list[np.ndarray], dict[str, float]]:
        assert (n > 0) and (n <= len(self.seeds))
        failures = []
        actions = []
        acc_rewards = []
        final_obs_list = []
        behaviors_list = []

        for i in range(n):
            failure, final_obs, action_seq, reward_seq, behaviors, _frames = (
                self._record_execution(
                    policy,
                    input,
                    record=False,
                    deterministic=deterministic,
                    seed=self.seeds[i],
                )
            )
            failures.append(failure)
            actions.append(action_seq)
            acc_rewards.append(np.sum(reward_seq))
            final_obs_list.append(final_obs)
            behaviors_list.append(np.array(list(behaviors.values())))

        # metrics for possible generic behavior space
        ep_length = [len(l) for l in actions]
        action_dist = compute_action_distributions(actions, range=[0, 5], bins=5)

        measures = dict(
            length_mean=np.mean(ep_length),
            length_std=np.std(ep_length),
            length_spread=max(ep_length) - min(ep_length),
            action_std=compute_action_std(actions),
            action_entropy=compute_entropy(action_dist),
            action_dist=action_dist
        )

        return np.mean(acc_rewards), np.mean(failures), final_obs_list, behaviors_list, measures

    def _record_execution(
        self,
        policy: DQN,
        positions: np.ndarray,
        record: bool = False,
        deterministic: bool = True,
        seed: int = None,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, dict[str, float], List[np.ndarray]]:
        if seed is None:
            seed = self.seeds[0]

        obs, _info = self.env.reset(seed=seed, options={"config": {"input": positions}})

        # obs_seq = []
        action_seq = []
        reward_seq = []
        # trajectory = []
        frames = []
        speeds = []

        terminated = truncated = False

        while not (terminated or truncated):
            if record:
                frames.append(self.env.render())

            action, _state = policy.predict(
                obs, state=None, deterministic=deterministic
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            # obs_seq.append(obs)
            action_seq.append(action)
            reward_seq.append(reward)
            speeds.append(info["speed"])
            # trajectory.append(
            #     np.array(self.env.get_wrapper_attr("controlled_vehicles")[0].position)
            # )


        if record:
            frames.append(self.env.render())

        xpos = [v.position[0] for v in self.env.get_wrapper_attr("road").vehicles]

        measures = {
            "mean_speed": np.mean(speeds),
            "num_overtaking": sorted(xpos).index(
                self.env.get_wrapper_attr("controlled_vehicles")[0].position[0]
                )
        }

        return (
            info["crashed"],
            obs.copy(),
            np.array(action_seq),
            np.array(reward_seq),
            measures,
            # np.vstack(trajectory),
            frames,
        )


### Debugging functions (different ways to execute episodes)


def debug_env(env: gym.Env, action: int = 1, seed: int = None, record: bool = False):
    """
    Resets `env` with `seed` only if the latter is not None.
    Debugging helper that performs `action` at each step until termination.
    Returns:
        - history (List[np.ndarray]): the list of the positions of all vehicles (shape of (num_vehicles, 2)) at each step.
        - rewards (np.ndarray): the reward received after each action.
        - velocities (np.ndarray): the speed of the ego vehicle.
        - info (Dict): the last `info` dictionnary.
        - frames (List[np.ndarray]): the list of `env.render()` if `record` is True; [] otherwise.
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    terminated = truncated = False

    history = [[np.array(v.position)] for v in env.get_wrapper_attr("road").vehicles]

    rewards = []
    frames = []
    velocities = []

    if record:
        frames.append(env.render())

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        velocities.append(info["speed"])
        if record:
            frames.append(env.render())
        for i, v in enumerate(env.get_wrapper_attr("road").vehicles):
            history[i].append(np.array(v.position))

    # print("============ debug_env print ============")
    # print("acc_reward", sum(rewards), "info:")
    # print(info)
    # print("============================")

    return (
        [np.vstack(h) for h in history],
        np.array(rewards),
        np.array(velocities),
        info,
        frames,
    )


def has_enough_initial_distance(input: np.ndarray, min_distance: float = 10.0):
    """Checks if the distance between the ego vehicle and the next vehicle in front of it is larger than `min_distance`."""
    ego_vehicle_index = np.argmin(input[:, 0])
    init_x, init_lane = input[ego_vehicle_index]

    closest_vehicle_index = 0
    init_dist = np.inf

    for i, data in enumerate(input):
        if (i != ego_vehicle_index) and (data[1] == init_lane):
            dist = data[0] - init_x
            if dist < init_dist:
                closest_vehicle_index = i
                init_dist = dist
                # print("closer vehicle found at {} ({})".format(closest_vehicle_index, init_dist))

    return init_dist >= min_distance


def fix_ego_position(input: np.ndarray):
    if has_enough_initial_distance(input):
        return input
    else:
        ego_vehicle_index = np.argmin(input[:, 0])
        fixed_input = input.copy()
        fixed_input[ego_vehicle_index][0] -= 10.0
        return fixed_input


if __name__ == "__main__":
    executor = HighwayTestManager(seeds=[None])

    rng = np.random.default_rng(0)
    model = executor.load_policy(
        "saved_models/dqnagent/checkpoint-35000.tar"
    )

    num_tests = 10

    inputs = np.array([executor.generate_input(rng) for _ in range(num_tests)])

    from pathlib import Path

    from my_highway import plot_input
    from plot import save_gif

    results_dir = Path("test_hw")
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_input(inputs[0])
    fig.savefig(results_dir / "test_input.png")

    mutants = []

    for i in range(5):
        i_dir = results_dir / "{}".format(i)
        i_dir.mkdir(parents=True, exist_ok=True)
        executor.seed = i
        failure, obs_seq, action_seq, reward_seq, trajectory, frames = (
            executor._record_execution(
                model,
                inputs[0],
                record=True,
                deterministic=True,
                seed=i
            )
        )

        print("========= RUN {} =========".format(i))
        print(failure, sum(reward_seq), len(reward_seq))
        save_gif(frames, i_dir / "exec.gif", duration=200)

        print("======= Mutation =======")
        mutant = executor.mutate_input(inputs[0], rng)
        mutants.append(mutant.copy())
        plot_input(mutant)[0].savefig(i_dir / "test_mutant.png")

        failure, obs_seq, action_seq, reward_seq, trajectory, frames = (
            executor._record_execution(model, mutant, record=True, deterministic=True, seed=i)
        )

        print(failure, sum(reward_seq), len(reward_seq))
        save_gif(frames, i_dir / "exec_mutant.gif", duration=200)
        print("===========================")

    print("=========== DONE! ==========")
    np.save(
        file=results_dir / "mutants.npy",
        arr=np.array(mutants),
        allow_pickle=True
    )