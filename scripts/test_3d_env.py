import diffmp


env = diffmp.problems.Environment.random(3, 10, 0.2, diffmp.problems.etc.Dim.THREE_D)

# for obstacle in env.obstacles:
#     print(obstacle.area())

print(f"{env.p_obstacles=}")

env.plot()
