from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
import torch
import yaml

import diffmp

model_path = Path("data/models/test.pt")


def out_to_mps(actions: npt.NDArray, states:npt.NDArray, out_index:int):
    length = actions.shape[0]
    mps = []
    for i in range(length):
        mp = {"time_stamp":0,
              "cost": 0.5,
              "feasible": 1,
              "traj_feas": 1,
              "goal_feas": 1,
              "start_feas": 1,
              "col_feas": 1,
              "x_bounds_feas": 1,
              "u_bounds_feas": 1,
              "start": states[i,:3].tolist(),
              "goal": states[i,-3:].tolist(),
              "max_jump":0,
              "max_collision": 0,
              "start_distance": 0,
              "goal_distance": 0,
              "x_bound_distance": 0,
              "u_bound_distance": 0,
              "num_states": 6,
              "num_actions": 5,
              "info" :"i:480-s:0-l:5",
              "actions": actions[i].reshape(-1, 2).tolist(),
              "states": states[i].reshape(-1, 3).tolist()}
        mps.append(mp)
    with open(f"out{out_index}.yaml", "w") as file:
        yaml.safe_dump(mps, file, default_flow_style=None)



def train():
    config = diffmp.torch.Config.from_yaml(Path("scripts/conf.yaml"))
    model = diffmp.torch.Model(config)
    model.path = model_path
    diffmp.torch.train(model, 1000)
    model.save()


def load():
    instance = diffmp.problems.Instance.from_dict(
        diffmp.utils.load_yaml(Path("../example/bugtrap.yaml"))
    )
    config = diffmp.torch.Config.from_yaml(Path("scripts/conf.yaml"))
    model = diffmp.torch.Model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    for out_index in range(10):
        out = diffmp.torch.sample(model, 100, instance).detach().numpy()
        actions = np.clip(out[:, :10], -0.5, 0.5)
        theta_0 = out[:, 10]
        q = np.zeros((actions.shape[0], 3))
        q[:, 2] = theta_0
        states_ls: List[npt.NDArray] = [q]
        for i in range(5):
            q = config.dynamics.step(q, actions[:, 2 * i : 2 * i + 2])
            states_ls.append(q)
        states = np.concatenate(states_ls, axis=1)
        out_to_mps(actions, states, out_index)
    



if __name__ == "__main__":
    train()
    load()
