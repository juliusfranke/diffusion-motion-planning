import yaml
from pathlib import Path

import diffmp
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with constrained layout
fig = plt.figure(figsize=(6, 6), constrained_layout=True)

gs = gridspec.GridSpec(2, 2, figure=fig)

# Create subplots in the grid
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[0, 1]),
]
# fig, axes = plt.subplots(1, 4)
# plt.subplots_adjust(wspace=0, hspace=0)

dynamics = "unicycle2_v0"
instance_0 = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap_2.yaml"))
# instance_1 = diffmp.problems.Instance.from_yaml(
#     Path("data/test_instances/unicycle2_v0/7.yaml")
# )
instance_1 = diffmp.problems.Instance.from_yaml(
    Path("data/test_instances/unicycle2_v0/37.yaml")
)
instance_2 = diffmp.problems.Instance.from_yaml(
    Path("data/test_instances/unicycle2_v0/12.yaml")
)
instances = [instance_0, instance_1, instance_2]
config = diffmp.utils.DEFAULT_CONFIG
# tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
N_MP = 100
composite_config = diffmp.torch.CompositeConfig.from_yaml(
    Path("data/models/unicycle2_v0.composite.yaml")
)
for i, instance in enumerate(instances):
    ax = axes[i]
    tmp_path = Path("data/output.yaml")
    diffmp.utils.export_composite(composite_config, instance, tmp_path, n_mp=N_MP)
    temp = config | {
        "mp_path": str(tmp_path),
    }
    task = diffmp.utils.Task(
        instance,
        temp,
        2000,
        4000,
        [],
    )
    task = diffmp.utils.execute_task(task)
    best_cost = np.inf
    states = np.zeros(shape=(2, 2))

    for solution in task.solutions:
        if solution.cost < best_cost:
            states = solution.optimized.states

    # solution = task.solutions[-1].optimized.states
    ax.plot(states[:, 0], states[:, 1], linewidth=2)
    instance.plot(ax)
with open("data/output.yaml", "r") as file:
    data = yaml.safe_load(file)

for _ in range(10):
    # states = np.array(data[i]["states"])
    states = np.array(random.choice(data)["states"])
    axes[3].plot(states[:, 0], states[:, 1])

axes[3].axis("off")
x_b = axes[3].get_xbound()
y_b = axes[3].get_ybound()
bounds_x = [x_b[0], x_b[1], x_b[1], x_b[0], x_b[0]]
bounds_y = [y_b[0], y_b[0], y_b[1], y_b[1], y_b[0]]
axes[3].plot(bounds_x, bounds_y, color="black")
for ax in axes:
    ax.set_adjustable("box")  # Ensures equal aspect ratio
    ax.set_aspect("auto")
plt.savefig("data/plots/combined.eps")
plt.show()
