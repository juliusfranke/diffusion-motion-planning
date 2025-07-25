import matplotlib.pyplot as plt
import yaml
import numpy as np
import random

with open("data/output.yaml", "r") as file:
    data = yaml.safe_load(file)

fig, ax = plt.subplots(1)
for _ in range(10):
    # states = np.array(data[i]["states"])
    states = np.array(random.choice(data)["states"])
    ax.plot(states[:, 0], states[:, 1])

ax.axis("off")
x_b = ax.get_xbound()
y_b = ax.get_ybound()
bounds_x = [x_b[0], x_b[1], x_b[1], x_b[0], x_b[0]]
bounds_y = [y_b[0], y_b[0], y_b[1], y_b[1], y_b[0]]
ax.plot(bounds_x, bounds_y, color="black")
plt.savefig("data/plots/sampled_mps.eps")
plt.show()
