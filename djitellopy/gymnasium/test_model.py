import numpy as np
from stable_baselines3 import PPO
from human_nav_env import HumanNavEnv

env = HumanNavEnv()
model = PPO.load("ppo_human_nav")

success_count = 0
total_steps = []
total_rewards = []

with open("test_log.txt", "w") as log_file:
    for ep in range(50):
        obs, _ = env.reset()
        ep_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            if steps % 5 == 0:
                log_str = (f"Ep {ep}, Step {steps}: Pos {env.drone_pos}, "
                          f"Obs {obs}, Action {action}, Reward {reward}, "
                          f"Distance {env._get_distance()}\n")
                print(log_str, end="")
                log_file.write(log_str)
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        if terminated:
            success_count += 1
        ep_str = f"Ep {ep}: Reward {ep_reward}, Steps {steps}, Success: {terminated}\n"
        print(ep_str, end="")
        log_file.write(ep_str)

    summary = (f"Avg Reward: {np.mean(total_rewards)}, "
              f"Avg Steps: {np.mean(total_steps)}, "
              f"Success Rate: {success_count / 50 * 100}%\n")
    print(summary)
    log_file.write(summary)