"""
File containing utility functions that do not need to be classes.
"""

import json
import warnings
import pickle
from typing import List

import numpy as np


def encode_input(x: np.ndarray):
    if x.shape != (21, 2):
        warnings.warn(f"Numpy array has a un-usual shape {x.shape}")
    return (
        str(x.flatten())
        .replace("\n", "")
        .replace("  ", "")
    )


def decode_input(x: str):
    return np.fromstring(x[1:-1], dtype=np.int32, sep=" ").reshape(-1, 2)


def decode_logs(x: str):
    """To be used with MDPFuzz's logs"""
    return np.fromstring(
        x.replace("[", "").replace("]", ""),
        dtype=np.int32,
        sep=","
    ).reshape(-1, 2)


def read_json_file(file_path: str):
    """From ChatGPT"""
    if not file_path.lower().endswith(".json"):
        raise ValueError("Invalid file extension. Expected a .json file.")

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
    except PermissionError:
        print(f"Error: Permission denied when trying to read the file '{file_path}'.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")

    return None


def write_json_file(data, file_path: str):
    """From ChatGPT"""
    if not file_path.lower().endswith(".json"):
        raise ValueError("Invalid file extension. Expected a .json file.")

    try:
        with open(file_path, "w") as file:
            json.dump(data, file)
    except PermissionError:
        print(
            f"Error: Permission denied when trying to write to the file '{file_path}'."
        )
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")


def write_pickle_file(file_path: str, data):
    if not file_path.lower().endswith(".pickle"):
        raise ValueError("Invalid file extension. Expected a .pickle file.")
    try:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    except PermissionError:
        print(
            f"Error: Permission denied when trying to write to the file '{file_path}'."
        )
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")


def read_pickle_file(file_path: str):
    if not file_path.lower().endswith(".pickle"):
        raise ValueError("Invalid file extension. Expected a .pickle file.")

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pickle.UnpicklingError:
        print(f"Error: The file '{file_path}' is not a unpicklable.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")

    return None


def pickle_to_txt(file_path: str, fmt: str = None):
    """
    Converts pickled data to a .txt file with Numpy.
    The pickled data is assumed to be a list of Numpy arrays.
    The latter are exported on a single line with `.reshape(1, -1)`.
    Note that the original shape is not saved.

    Paramters
    ---------
    fmt : str, optional
        - Format for `np.savetxt` (default to None).
    """
    pickle_obj = read_pickle_file(file_path)
    if not isinstance(pickle_obj, List) and not np.all(
        [isinstance(arr, np.ndarray) for arr in pickle_obj]
    ):
        raise TypeError("The pickled data is expected to be a list of Nympy arrays.")

    txt_file = open(file_path.split(".pickle")[0] + ".txt", "w")

    for arr in pickle_obj:
        if fmt is not None:
            np.savetxt(txt_file, arr.reshape(1, -1), delimiter=",", fmt=fmt)
        else:
            np.savetxt(txt_file, arr.reshape(1, -1), delimiter=",")

    txt_file.close()
