import os
os.makedirs("./logs/", exist_ok=True)
os.makedirs("./tensorboard/", exist_ok=True)

from human_nav_env import HumanNavEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

env = DummyVecEnv([lambda: HumanNavEnv()])

model = PPO(
    "MlpPolicy", env, verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log="./tensorboard/"
)

eval_callback = EvalCallback(
    env, best_model_save_path="./logs/",
    log_path="./logs/", eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True
)

model.learn(total_timesteps=100000, callback=eval_callback)
model.save("ppo_human_nav")