# import diffmp
from pathlib import Path
from meshlib.mrmeshpy import BooleanOperation, boolean
import diffmp.problems as pb
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import colors

from diffmp.problems.etc import plot_3dmesh, plot_3dmesh_to_2d
import diffmp
import dbcbs_py

dynamics = "unicycle1_v0"
data = {
    "delta_0": 0.5,
    "delta_rate": 0.9,
    "num_primitives_0": 200,
    "num_primitives_rate": 1.5,
    "alpha": 0.5,
    "filter_duplicates": True,
    "heuristic1": "reverse-search",
    "heuristic1_delta": 1.0,
    "mp_path": f"../new_format_motions/{dynamics}/{dynamics}.msgpack",
}


# env = pb.Environment.random(3, 10, 0.2, pb.Dim.THREE_D)
# ins = pb.Instance.random(3, 6, 0.2, ["integrator2_3d_v0"], dim= pb.Dim.THREE_D)
ins = pb.Instance.random(3, 6, 0.2, ["unicycle1_v0"], dim=pb.Dim.TWO_D)
ins.to_yaml(Path("test.yaml"))
breakpoint()
task = diffmp.utils.Task(
    ins,
    data,
    3000,
    3000,
    [],
)
task = diffmp.utils.execute_task(task)
breakpoint()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
blocked = boolean(
    ins.environment.blocked_mesh, ins.environment.boundary_mesh, BooleanOperation.Union
).mesh
plt.show()
# breakpoint()
sd = ins.environment.discretize_sd(n=20)
print(sd.shape)
fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax = [ax1, ax2, ax3]
# fig, ax = plt.subplots(ncols=2)

plot_3dmesh(blocked, ax=ax[2])
ins.plot(ax=ax[0])
# for i in range(15):
#     for j in range(15):
#         c = intersection_matrix[j,i]
#         ax.text(i, j, str(c), va='center', ha='center')
# sd = np.clip(sd, 0, np.inf)
ax[1].matshow(sd[:, :, 0].T, vmin=-1, vmax=1)
ax[1].yaxis.set_inverted(False)
# plt.imshow(sd, cmap='hot', interpolation='nearest')
plt.show()


# print(f"{env.p_obstacles=}")

# breakpoint()

# env.plot()
