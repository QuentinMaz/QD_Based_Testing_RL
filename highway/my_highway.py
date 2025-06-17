from typing import Callable, List, Optional

import gymnasium as gym
import highway_env
import matplotlib.colors as mcolors
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from matplotlib import pyplot as plt

XKCD_COLORS = ["purple", "green", "blue", "pink", "red", "brown"]
INITIAL_SPEED = 23.0

NUM_LANES = 3
NUM_VEHICLES = 20
# validity threshold (from empirical evaluation based on study of HighwayEnv's collision handling)
X_COLLISION_THRESHOLD = 5.0
X_NEIGHBORHOOD_DISTANCE_THRESHOLD = 20.0
DENSITY_THRESHOLD = 4
MUTATION_INTENSITY = 10.0


### COLORING UTILITY FUNCTIONS


def get_xkcd_colors(color_names: List[str], to_int: Callable = None):
    """
    Return a list of xkcd colors.
    If `to_int` is None, the colors are tuples of floating points between 0 and 1.
    If not, the values are converted to integer values between 0 and 256 with the given cast function.
    """
    hex_colors = [mcolors.XKCD_COLORS["xkcd:{}".format(name)] for name in color_names]
    rgb_colors = [mcolors.to_rgb(c) for c in hex_colors]
    if to_int is not None:
        rgb_colors = [tuple([to_int(255 * v) for v in c]) for c in rgb_colors]

    return rgb_colors


class InitializableHighwayEnv(HighwayEnvFast):
    def __init__(
        self,
        config: dict = None,
        render_mode: Optional[str] = None,
        input: np.ndarray = None,
    ) -> None:
        self.input = input
        # sorted 2d positions defined by `input`
        if input is not None:
            self.vehicles_features = self._get_vehicles_features(input)
        else:
            self.vehicles_features = None

        # TODO: debugging code to eventually remove
        # defines somes random colors to visualize the first vehicles
        self.colors = get_xkcd_colors(XKCD_COLORS, np.int64)

        # init executes reset, which executes _reset() (involving the creation of the road and vehicles)
        super().__init__(config, render_mode)
        # sanity checking: no vehicle should be collided initially
        # if not self._is_valid(SANITY_CHECK_ALL_VEHICLES):
        #     raise ValueError('some vehicles are initially collided.')

    def _update(self, new_input: np.ndarray):
        self.input = new_input
        self.vehicles_features = self._get_vehicles_features(self.input)

    def _set_positions(self):
        # ego vehicle
        vehicle = Vehicle.create_random(
            self.road, speed=INITIAL_SPEED, lane_id=int(self.vehicles_features[0][1])
        )
        vehicle.position[0] = self.vehicles_features[0][0]
        vehicle = self.action_type.vehicle_class(
            self.road, vehicle.position, vehicle.heading, vehicle.speed
        )
        self.controlled_vehicles = [vehicle]
        # print("vehicle placed in", vehicle.position)
        self.road.vehicles.append(vehicle)

        # adds then the remaining vehicles
        for i in range(1, len(self.vehicles_features)):
            # uses copies?
            vehicle_feature = self.vehicles_features[i]

            vehicle_class = utils.class_from_path(
                "highway_env.vehicle.behavior.IDMVehicle"
            )  # type: IDMVehicle

            # lane = self.road.network.get_lane(("0", "1", int(vehicle_feature[1])))

            # v = vehicle_class(
            #     road=self.road,
            #     position=lane.position(vehicle_feature[0], 0),
            #     heading=0,
            #     speed=INITIAL_SPEED,
            # )
            v = vehicle_class.create_random(
                self.road, INITIAL_SPEED, lane_id=int(vehicle_feature[1])
            )

            v.position[0] = vehicle_feature[0]

            # print("vehicle placed in", v.position)
            self.road.vehicles.append(v)

        # TODO: debugging code to eventually remove
        for i, v in enumerate(self.road.vehicles[1 : len(self.colors)]):
            v.color = self.colors[i]

    def _create_vehicles(self) -> None:
        """Populates the road with vehicles according to the input."""
        # checks if an new initial situation has been passed in `options` from `reset()`
        if "input" in self.config:
            # print("Input detected in options when reset()!")
            new_input = self.config["input"]
            if not np.array_equal(self.input, new_input):
                # print(
                #     "The input provided is different than current one. (calling _update()...)"
                # )
                self._update(new_input)

        if self.input is not None:
            # print("self.input is not None. Setting the positions accordingly.")
            self._set_positions()
        else:
            # print("self.input is None. Random initialization.")
            # uses the usual random initialization if the environment has not been given any initial positions
            super()._create_vehicles()
            # initializes the input attribute based on the vehicles' positions
            x_pos = np.array([v.position[0] for v in self.road.vehicles])
            lane_indices = np.array([v.lane_index[-1] for v in self.road.vehicles])
            self._update(np.stack([x_pos, lane_indices], axis=1))
            # applies the current initial settings, i.e: changing all vehicles' speed to `INITIAL_SPEED`
            for v in self.road.vehicles:
                v.speed = INITIAL_SPEED
            # print("======== INPUT =========")
            # print(self.input)
            # print("========================")

    def _get_vehicles_features(self, input: np.ndarray) -> np.ndarray:
        """Sorts the 2d positions of `input`."""
        # seems to build a copy
        return input[input[:, 0].argsort()]

    def _get_collisions(self, check_all_vehicles=True):
        """Calls `handle_collisions` of the vehicles with `dt=0` and returns their crashed attributes."""
        dt = 0.0
        if check_all_vehicles:
            for i, vehicle in enumerate(self.road.vehicles):
                for other in self.road.vehicles[i + 1 :]:
                    vehicle.handle_collisions(other, dt)
                for other in self.road.objects:
                    vehicle.handle_collisions(other, dt)
        else:
            # checks only collisions for the ego vehicle
            for other in self.road.vehicles[1:]:
                self.vehicle.handle_collisions(other, dt)
            for other in self.road.objects:
                self.vehicle.handle_collisions(other, dt)

        return [v.crashed for v in self.road.vehicles]


### Additional utility functions


def compare_positions(positions: np.ndarray, input: np.ndarray):
    """Compares the two input assuming that they are initial positions to use with `InitializableHighwayEnv`."""
    scale_factor = 4.0

    # positions = np.vstack([
    #     np.array(v.position) # I guess it copies the values
    #     for v in env.get_wrapper_attr("road").vehicles
    # ])

    if positions.shape != input.shape:
        print("Not same shape: {} vs {}.".format(positions.shape, input.shape))
        return False

    # sorts the 2d positions
    positions = positions[positions[:, 0].argsort()]
    input = input[input[:, 0].argsort()]

    if not np.array_equal(positions[:, 0], input[:, 0]):
        print("Not the same the x positions.")
        return False

    if not np.array_equal(positions[:, 1], scale_factor * input[:, 1]):
        print("Not the same the y positions.")
        return False

    return True


def check_environment(env: gym.Env, input: np.ndarray):

    positions = np.vstack(
        [
            np.array(v.position)  # I guess it copies the values
            for v in env.get_wrapper_attr("road").vehicles
        ]
    )

    return compare_positions(positions, input)


### Input generation and mutation functions


def is_valid_x(input: np.ndarray):
    """Checks whether there is no vehicles horizontally too close between each other."""
    for lane in range(NUM_LANES):
        # gets the features whose lane matches
        x_pos = np.array([f[0] for f in input if f[1] == lane])
        x_pos.sort()
        for i in range(len(x_pos) - 1):
            if (x_pos[i + 1] - x_pos[i]) <= X_COLLISION_THRESHOLD:
                return False
        # print("LANE {} OK.".format(lane))
    return True


def is_valid_density(input: np.ndarray):
    """Density checking throught the number of neighbors of each point."""
    x_positions = input[:, 0]
    valid = [
        np.sum(np.abs(x_positions - x_positions[i]) < X_NEIGHBORHOOD_DISTANCE_THRESHOLD)
        - 1
        <= DENSITY_THRESHOLD
        for i in range(len(x_positions))
    ]
    # print("Density check", valid)
    return all(valid)


def generate_input(
    rng: np.random.Generator,
    num_vehicles: int = NUM_VEHICLES,
    num_lanes: int = NUM_LANES,
    num_attempts: int = 100,
):
    """Returns an input of shape (num_vehicles + 1, 2) whose the latter dimension describes the x position (world coordinates) and the index of the lane."""
    for i in range(num_attempts):
        x_pos = rng.integers(low=200, high=700, size=(num_vehicles + 1))
        lane_indices = rng.integers(low=0, high=num_lanes, size=(num_vehicles + 1))
        input = np.stack([x_pos, lane_indices], axis=1)
        if is_valid_x(input) and is_valid_density(input):
            ego_index = input[:, 0].argmin()
            input[ego_index, 0] -= 10
            return input.astype(np.int32)
    return None


def generate_even_input(
    rng: np.random.Generator,
    num_vehicles: int = NUM_VEHICLES,
    num_lanes: int = NUM_LANES,
    distance_between_vehicles: int = 25,
):
    """Returns an input of shape (num_vehicles + 1, 2) whose the latter dimension describes the x position (world coordinates) and the index of the lane."""
    x_pos = np.arange(
        start=200,
        stop=200 + (num_vehicles + 1) * distance_between_vehicles,
        step=distance_between_vehicles,
    ) + rng.integers(low=0, high=5, size=(num_vehicles + 1))
    x_pos[0] -= 10
    lane_indices = rng.integers(low=0, high=num_lanes, size=(num_vehicles + 1))
    input = np.stack([x_pos, lane_indices], axis=1)
    return input.astype(np.int32)


def mutate_positions(
    x: np.ndarray,
    rng: np.random.Generator,
    num_attempts: int = 100,
    initial_distance: float = 25,
) -> np.ndarray:
    for i in range(num_attempts):
        mutant = x.copy()
        mutant[:, 0] = np.clip(rng.normal(mutant[:, 0], 10.0), 200, 700)
        if is_valid_x(mutant) and is_valid_density(mutant):
            ego_index, next_vehicle_index = mutant[:, 0].argsort()[:2]
            if (
                mutant[next_vehicle_index, 0] - mutant[ego_index, 0]
            ) < initial_distance:
                # before = mutant[ego_index, 0]
                mutant[ego_index, 0] = mutant[next_vehicle_index, 0] - initial_distance
                # print("Ego vehicle moved from {} to {}.".format(before, mutant[ego_index, 0]))
                # print("Ego vehicle moved to {} from the next vehicle.".format(initial_distance))
            return mutant.astype(np.int32)
    print(
        "WARNING: unsuccessful positions mutation (copy of the input array returned)."
    )
    return x.copy().astype(np.int32)


def plot_input(input: np.ndarray, extend: bool = True):
    input_sorted = input[input[:, 0].argsort()]
    colors = (
        [(0.0, 0.0, 0.0)]
        + get_xkcd_colors(XKCD_COLORS)
        + [(0.0, 0.5, 0.9) for _ in range(len(input) - (1 + len(XKCD_COLORS)))]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(input_sorted[:, 0], input_sorted[:, 1], c=colors)
    if extend:
        ax.set_xlim(min(180, min(input[:, 0])), 1300)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "collision_reward": -10.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.0,
        "reward_speed_range": [23, 28],  # last in date
        "normalize_reward": False,
        "offscreen_rendering": True,
    }
    check_basic_initialization = False
    check_random_initialization = False
    check_mutation = False
    record_multiple_runs = True

    if check_basic_initialization:
        rng = np.random.default_rng(0)
        input = generate_input(rng)
        print(input)
        print("x validity", is_valid_x(input))
        print("density validity", is_valid_density(input))

        fig, ax = plot_input(input)
        fig.savefig("hw_test_env_init_input.png")

        env = InitializableHighwayEnv(
            config=config, render_mode="rgb_array", input=input
        )
        crashes = env._get_collisions()
        print("crashes?", crashes)
        print("valid?", any(crashes) == False)

        print("num vehicles", len(env.road.vehicles))

        fig, ax = plt.subplots()
        ax.imshow(env.render())
        fig.tight_layout()
        fig.savefig("hw_test_env_init.png")
        print("Env. set with input?", check_environment(env, input))
        env.close()

    if check_random_initialization:
        env = InitializableHighwayEnv(config=config, render_mode="rgb_array")
        crashes = env._get_collisions()
        print("crashes?", crashes)
        print("valid?", any(crashes) == False)

        print("num vehicles", len(env.road.vehicles))
        input = env.input.copy()
        print(input)
        fig, ax = plot_input(input, extend=False)
        fig.savefig("hw_test_env_random_init_input.png")

        print("Env. set with input?", check_environment(env, input))

        fig, ax = plt.subplots()
        ax.imshow(env.render())
        fig.tight_layout()
        fig.savefig("hw_test_env_random_init.png")
        env.close()

        # checks that the initial speeds have been set
        print(
            "Initial speed (re)set?",
            np.all(
                [
                    v.speed == INITIAL_SPEED
                    for v in env.get_wrapper_attr("road").vehicles
                ]
            ),
        )

    if check_mutation:
        rng = np.random.default_rng(0)
        input = generate_input(rng)
        print("============= INPUT =============")
        print(input)
        print("x validity", is_valid_x(input))
        print("density validity", is_valid_density(input))
        print("============================")

        mutant = mutate_positions(input, rng)

        print("============= MUTANT =============")
        print(mutant)
        print("x validity", is_valid_x(mutant))
        print("density validity", is_valid_density(mutant))
        print("============================")

        print("============= CHECKING AT ENV. LEVEL (with mutant) =============")
        env = InitializableHighwayEnv(
            config=config, render_mode="rgb_array", input=mutant
        )
        crashes = env._get_collisions()
        print("crashes?", crashes)
        print("valid?", any(crashes) == False)

    if record_multiple_runs:
        rng = np.random.default_rng(0)
        input = generate_input(rng)
        env = InitializableHighwayEnv(
            config=config, render_mode="rgb_array", input=input
        )

        from executor import debug_env
        from plot import save_gif

        seeds = np.arange(3).tolist()
        for s in seeds:
            history, rewards, velocities, info, frames = debug_env(
                env, action=1, seed=s, record=True
            )
            save_gif(frames, "hw_test_env_debug_{}.gif".format(s), duration=200)

    print("Done.")

    for i in range(3):
        obs, info = env.reset()
        fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
        print(
            env.get_wrapper_attr("road").vehicles[0].speed,
            env.get_wrapper_attr("road").vehicles[0].position,
        )
        frames = [env.render()]
        env.step(3)
        print(
            env.get_wrapper_attr("road").vehicles[0].speed,
            env.get_wrapper_attr("road").vehicles[0].position,
        )
        frames.append(env.render())
        env.step(3)
        frames.append(env.render())
        env.step(1)
        print(
            env.get_wrapper_attr("road").vehicles[0].speed,
            env.get_wrapper_attr("road").vehicles[0].position,
        )
        frames.append(env.render())
        for f, ax in zip(frames, axs):
            ax.imshow(f)
        fig.tight_layout()
        fig.savefig("test_myenv_{}.png".format(i))
        print("====================")
