from typing import List

import numpy as np
from scipy.stats import entropy


def compute_action_std(actions_list: List[np.ndarray], max_time: int = None) -> float:
    size = min([len(l) for l in actions_list])
    if max_time is not None:
        size = min(size, max_time)

    action_std = np.mean(np.std([sub_list[:size] for sub_list in actions_list], axis=0))
    return action_std


def compute_action_distributions(
    actions_list: List[np.ndarray],
    range: List[int],
    bins: int,
    epsilon: float = 1e-5,
    max_time: int = None,
) -> np.ndarray:
    size = min([len(l) for l in actions_list])
    if max_time is not None:
        size = min(size, max_time)
    sub_action_list = [
        sub_list[:size] for sub_list in actions_list
    ]  # type: List[np.ndarray]
    # shape (time, action, samples)
    action_values = np.stack(sub_action_list, axis=-1)
    # bins every action
    action_distribution = np.apply_along_axis(
        func1d=lambda x: np.histogram(x, bins=bins, range=range, density=True)[0],
        arr=action_values,
        axis=-1,  # 2
    )
    action_distribution += epsilon
    normalized_distribution = np.apply_along_axis(
        func1d=lambda x: x / sum(x), arr=action_distribution, axis=-1  # 2
    )
    return normalized_distribution


def compute_entropy(distributions: np.ndarray) -> np.ndarray:
    return (np.log(distributions) * -distributions).sum(axis=-1)


def compute_divergence_time(actions_list: List[np.ndarray], range: List[int], bins: int) -> int:
    size = min([len(l) for l in actions_list])
    sub_action_list = [sub_list[:size] for sub_list in actions_list]  # type: List[np.ndarray]
    action_values = np.stack(sub_action_list, axis=-1)
    binned_actions = np.apply_along_axis(
        func1d=lambda x: np.histogram(x, bins=bins, range=range, density=False)[0],
        arr=action_values,
        axis=-1
    )
    # first time (argmax) for which the sums of the binned actions do not equal to 1 (i.e., the actions differ)
    # actions are binned first since they can be continuous (e.g., Bipedal Walker)
    return ((binned_actions != 0).sum(axis=-1) != 1).any(axis=-1).argmax()


if __name__ == "__main__":
    bw_actions = [np.random.rand(300, 4) for _ in range(3)]
    ll_actions = [np.random.randint(0, 4, size=(775, 1)) for _ in range(3)]
    hw_actions = [np.random.randint(0, 5, size=(30, 1)) for _ in range(3)]

    # testing distribution computation
    ll_dist = compute_action_distributions(ll_actions, range=[0, 4], bins=4)
    assert ll_dist.shape == (775, 1, 4)
    assert np.allclose(np.sum(ll_dist, axis=2), 1)

    bw_dist = compute_action_distributions(bw_actions, range=[-1, 1], bins=10)
    assert bw_dist.shape == (300, 4, 10)
    assert np.allclose(np.sum(bw_dist, axis=2), 1)

    hw_dist = compute_action_distributions(hw_actions, range=[0, 5], bins=5)
    assert hw_dist.shape == (30, 1, 5)
    assert np.allclose(np.sum(hw_dist, axis=2), 1)

    # testing entropy
    ll_entropy = compute_entropy(ll_dist).mean()
    bw_entropy = compute_entropy(bw_dist).mean()
    hw_entropy = compute_entropy(hw_dist).mean()

    print("LL entropy:", ll_entropy)
    print("BW entropy:", bw_entropy)
    print("HW entropy:", hw_entropy)
    assert np.allclose(ll_entropy, entropy(ll_dist, axis=-1).mean())
    assert np.allclose(bw_entropy, entropy(bw_dist, axis=-1).mean())
    assert np.allclose(hw_entropy, entropy(hw_dist, axis=-1).mean())
