from diffmp.dynamics.base import DynamicsBase
from diffmp.dynamics.unicycle_1 import UnicycleFirstOrder
from icecream import ic
import numpy as np


def test():
    dt = 0.1
    u = ["s", "phi"]
    q = ["x", "y", "theta"]
    u_lims = {"lower": -0.5, "upper": 0.5}
    u_lims_2 = {"lower": {"s": -0.5, "phi": -0.5}, "upper": {"s": 0.5, "phi": 0.5}}

    a = UnicycleFirstOrder(dt=0.1, u_lims=u_lims)
    b = DynamicsBase(dt=0.1, q=q, u=u, u_lims=u_lims_2)
    ic(str(a))
    ic(a._q_lims)
    ic(a._u_lims)
    q1 = a.step(q=np.array([0, 0, 0]), u=np.array([0.3, 0.1]))
    q2 = a.step(q=np.array([[0, 0, 0], [1, 2, 3]]), u=np.array([[0, 0.5], [-0.1, 0.3]]))
    ic(q1)
    ic(q2)
    # ic(b._q_lims)
    # ic(b._u_lims)
