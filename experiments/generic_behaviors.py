import tqdm, os, pickle, warnings, sys

import numpy as np

import pandas as pd


SEED = 5
ENV_SEED = 0
N = 3
NUM_INPUTS = 10


if __name__ == "__main__":
    use_case = "ll" if len(sys.argv) == 1 else sys.argv[1]
    assert use_case in ["bw", "ll"]

    DF_FILENAME = f"{use_case}_behavior_data"
    rng = np.random.default_rng(seed=SEED) # type: np.random.Generator

    if use_case == "ll":
        from ll_common import load_lunar_lander_model, execute_stochastic_policy
        model = load_lunar_lander_model()
        inputs = rng.uniform(low=[-1000, -1000], high=[1000, 0], size=(NUM_INPUTS, 2))
        columns =  ["x", "y"]
        encode_input = lambda x: [x[0], x[1]]
        sim_steps = 1000
    else:
        from bw_common import load_model, generate_inputs, execute_stochastic_policy
        model = load_model()
        inputs = generate_inputs(rng, NUM_INPUTS)
        columns =  ["str_input"]
        encode_input = lambda x: [str(x)]
        sim_steps = 300

    # dataframe storage
    columns += ["reward_mean", "failure_prob", "length_mean", "length_std", "length_spread", "action_std", "action_entropy"]
    data = {
        k: [] for k in columns
    }

    features_list = []

    for input in tqdm.tqdm(inputs):
        mean_reward, failure_prob, features, _final_obs, behaviors = execute_stochastic_policy(input, model, ENV_SEED, N, sim_steps=sim_steps)

        record = encode_input(input) + [mean_reward, failure_prob] + [behaviors.get(c) for c in columns if c in list(behaviors.keys())]
        [data[k].append(v) for k, v in zip(columns, record)]

        features_list.append(features)

    df = pd.DataFrame.from_dict(data, orient="columns")
    csv_filename = DF_FILENAME + ".csv"
    df.to_csv(csv_filename, index=0, mode="a", header=(not os.path.exists(csv_filename)))

    np_file = DF_FILENAME + "_features.pickle"
    if not os.path.exists(np_file):
        warnings.warn("No pickle file found to add the discovered features.", UserWarning)
        with open(np_file, "wb") as file:
            pickle.dump(
                np.vstack(features_list), file
            )
    else:
        # reads pickle directly
        features = np.load(np_file, allow_pickle=True)
        warnings.warn(f"Found {len(features)} features.")
        # but saves with pickle
        with open(np_file, "wb") as file:
            pickle.dump(
                np.vstack([features] + features_list), file
            )