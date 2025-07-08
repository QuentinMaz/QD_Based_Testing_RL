import tqdm, os, pickle, warnings, sys

import numpy as np

import pandas as pd

from bw_framework import BWFramework
from common import ENV_SEEDS, MEASURES, load_lunar_lander_model, load_bipedal_walker_model
from ll_framework import LLFramework


SEED = 5
NUM_INPUTS = 10


if __name__ == "__main__":
    use_case = "ll" if len(sys.argv) == 1 else sys.argv[1]
    assert use_case in ["bw", "ll"]

    DF_FILENAME = f"{use_case}_2"
    rng = np.random.default_rng(seed=SEED) # type: np.random.Generator
    executor_cls = LLFramework if use_case == "ll" else BWFramework
    if use_case == "ll":
        model = load_lunar_lander_model()
        inputs = rng.uniform(low=[-1000, -1000], high=[1000, 0], size=(NUM_INPUTS, 2))
        columns =  ["x", "y"]
        encode_input = lambda x: [x[0], x[1]]
    else:
        model = load_bipedal_walker_model()
        columns =  ["str_input"]
        encode_input = lambda x: [str(x)]

    descriptors = ["action_entropy", "length_spread"]
    executor = executor_cls(ENV_SEEDS, 50, MEASURES, MEASURES)


    # dataframe storage
    columns += ["reward_mean", "failure_prob", "length_mean", "length_std", "length_spread", "action_std", "action_entropy"]
    data = {
        k: [] for k in columns
    }

    inputs = executor.generate_inputs(NUM_INPUTS)
    action_densities = []
    action_entropies = []
    for input in tqdm.tqdm(inputs):
        mean_reward, failure_prob, final_obs_list, behaviors_list, measures = executor.execute_stochastic_policy(input, model, ENV_SEEDS)

        record = encode_input(input) + [mean_reward, failure_prob] + [measures.get(c) for c in columns if c in MEASURES]
        [data[k].append(v) for k, v in zip(columns, record)]

        # raw densities of the action distributions
        action_dist = measures["action_dist"]
        action_densities.append(action_dist)
        # and the densities' entropy
        action_entropies.append(
            (np.log(action_dist) * -action_dist).sum(axis=-1)
        )

    df = pd.DataFrame.from_dict(data, orient="columns")
    csv_filename = DF_FILENAME + ".csv"
    df.to_csv(csv_filename, index=0, header=(not os.path.exists(csv_filename)))

    for l, suffix in zip([action_densities, action_entropies], ["densities", "entropies"]):
        np_file = DF_FILENAME + f"_action_{suffix}.pickle"
        with open(np_file, "wb") as file:
            pickle.dump(
                l, file
            )