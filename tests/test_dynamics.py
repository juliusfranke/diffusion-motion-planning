from diffmp.dynamics import get_dynamics
from icecream import ic
import numpy as np
import yaml


def test():
    with open("unicycle1_v0.yaml", "r") as file:
        data = yaml.safe_load(file)

    with open("unicycle2_v0.yaml", "r") as file:
        data_2 = yaml.safe_load(file)

    a = get_dynamics(data)
    b = get_dynamics(data_2)
    u= np.array([0.3,0.5])
    q = np.array([0,0,0])
    q2 = np.array([0,0,0,0,0])
    u2 = np.array([0.1,0.1])
    for i in range(55):
        # q = a.step(q=q, u=u)
        q2 = b.step(q=q2, u=u2)
        print(q2)

