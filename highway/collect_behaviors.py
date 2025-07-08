import pickle
import tqdm, os

import numpy as np

import pandas as pd


SEED = 5
ENV_SEEDS = [0, 1, 2]
NUM_INPUTS = 10


if __name__ == "__main__":
    DF_FILENAME = "data"
    rng = np.random.default_rng(seed=SEED) # type: np.random.Generator

    from executor import HighwayTestManager
    from my_utils import encode_input

    executor = HighwayTestManager(seeds=ENV_SEEDS)
    model = executor.load_policy()
    inputs = executor.generate_inputs(rng, NUM_INPUTS)

    columns =  ["str_input"]
    # dataframe storage
    columns += ["reward_mean", "failure_prob", "length_mean", "length_std", "length_spread", "action_std", "action_entropy"]
    data = {
        k: [] for k in columns
    }

    action_densities = []
    action_entropies = []

    for input in tqdm.tqdm(inputs):
        mean_reward, failure_prob, final_obs_list, behaviors_list, measures = executor.execute_stochastic_policy(
            input,
            model,
            len(ENV_SEEDS),
            deterministic=True
        )

        record = [encode_input(input), mean_reward, failure_prob] + [measures.get(c) for c in columns if c in list(measures.keys())]
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