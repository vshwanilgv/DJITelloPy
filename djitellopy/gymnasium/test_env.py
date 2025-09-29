# from human_nav_env import HumanNavEnv

# env = HumanNavEnv()
# obs, _ = env.reset()
# print("Initial obs:", obs)

# for _ in range(10):
#     action = env.action_space.sample()  # Random
#     obs, reward, term, trunc, _ = env.step(action)
#     env.render()
#     print(f"Action: {action}, Obs: {obs}, Reward: {reward}")
#     if term or trunc:
#         break

from human_nav_env import HumanNavEnv
env = HumanNavEnv()
obs, _ = env.reset()
for _ in range(10):
    obs, r, t, tr, _ = env.step(0)  # Forward
    print(f"Pos {env.drone_pos}, Obs {obs}, Reward {r}, Distance {env._get_distance()}, Terminated {t}")