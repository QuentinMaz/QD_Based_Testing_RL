from typing import Union

import numpy as np
import torch
from rl_agents.agents.common.factory import load_agent, load_environment

# DQNAgent <- AbstractDQNAgent <- StochasticAgent <- AbstractAgent
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
from stable_baselines3 import DQN

from device import DEVICE


AGENT_CONFIG = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "EgoAttentionNetwork",
        "layers": [256, 256],
        "embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7,
        },
        "others_embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7,
        },
        "self_attention_layer": None,
        "attention_layer": {"type": "EgoAttention", "feature_size": 64, "heads": 2},
        "output_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
        },
    },
    "double": True,
    "loss_function": "l2",
    "optimizer": {"lr": 0.0005},
    "gamma": 0.99,
    "n_steps": 1,
    "batch_size": 64,
    "memory_capacity": 15000,
    "target_update": 512,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6000,
        "temperature": 1.0,
        "final_temperature": 0.05,
    },
}


ENV_CONFIG = {
    "id": "highway-fast-v0",
    "import_module": "highway_env",
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
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


class AgentWrapper:
    """Wrapper class for using agents from rl_agents as SB3 ones."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.device = self.agent.device
        # should not be necessary
        try:
            self.agent.eval()
        except:
            pass

    def predict(
        self, observations: np.ndarray, state=None, deterministic=True, to_numpy=True
    ):
        # batches single observation
        if len(observations.shape) == 2:
            x = torch.as_tensor(observations[None], dtype=torch.float).to(self.device)
        else:
            x = torch.as_tensor(observations, dtype=torch.float).to(self.device)

        with torch.no_grad():
            qvalues = self.agent.value_net(
                torch.as_tensor(x, dtype=torch.float).to(self.device)
            )  # type: torch.Tensor
            _max_qvalues, actions = qvalues.max(1)

        if to_numpy:
            return actions.cpu().numpy(), None
        else:
            return actions, None


def load_dqn(path: str):
    model = DQN.load(path, device=DEVICE)
    return model


def load_dqnagent(path: str):
    env = load_environment(ENV_CONFIG)
    agent = load_agent(AGENT_CONFIG, env)  # type: DQNAgent
    env.close()
    agent.load(path)
    return AgentWrapper(agent)


MODEL_DICT = {"DQNAgent": load_dqnagent, "DQN": load_dqn}


def load_model(model_path: str, model_type: str) -> Union[DQN, AgentWrapper]:
    if not model_type in MODEL_DICT.keys():
        raise ValueError("Agent {} not available for loading.".format(model_type))
    return MODEL_DICT[model_type](model_path)