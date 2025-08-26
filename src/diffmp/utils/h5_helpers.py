import h5py
import numpy as np

def get_array(group: h5py.Group, name: str = "scalars") -> np.ndarray:
    dset = group[name]
    if not isinstance(dset, h5py.Dataset):
        raise TypeError(f"{name} is not a dataset")
    return dset[()]


def get_string_array(group: h5py.Group, name: str) -> list[str]:
    dset = group[name]
    if not isinstance(dset, h5py.Dataset):
        raise TypeError(f"{name} is not a dataset")

    arr: np.ndarray = dset[()]  # dtype is usually 'Sxx' or object
    flat: np.ndarray = arr.ravel()

    return [b.decode("utf-8") for b in flat]


def get_columns(group: h5py.Group, name: str = "columns") -> list[tuple[str, str]]:
    dset = group[name]
    if not isinstance(dset, h5py.Dataset):
        raise TypeError(f"{name} is not a dataset")

    arr: np.ndarray = dset[()]  # shape (D, 2), dtype='S'
    cols: list[tuple[str, str]] = []
    for pair in arr:
        col_tuple = tuple(
            s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
            for s in pair
        )
        assert len(col_tuple) == 2
        cols.append(col_tuple)
    return cols


def load_environment_dataset(
    path: str,
) -> tuple[np.ndarray, list[str], list[str], list[tuple[str, str]]]:
    with h5py.File(path, "r") as f:
        environments = get_string_array(f, "environments")

        l5 = f["l5"]
        if not isinstance(l5, h5py.Group):
            raise TypeError
        scalars = get_array(l5, "scalars").astype("float32")
        env_ids = get_string_array(l5, "env_ids")
        cols = get_columns(l5, "columns")

    return scalars, env_ids, environments, cols
