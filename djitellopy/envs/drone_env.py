# envs/drone_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneFollowEnv(gym.Env):
    """Custom Drone Following Environment for Gymnasium"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Action space: 0-8 (9 discrete actions)
        # 0=forward, 1=back, 2=left, 3=right, 4=up, 5=down, 6=rotate_cw, 7=rotate_ccw, 8=hover
        self.action_space = spaces.Discrete(9)

        # Observation space: [dx, dy, bbox_size, obstacle_flag]
        # dx, dy ∈ [-1, 1], bbox_size ∈ [0, 1], obstacle_flag ∈ {0, 1}
        low = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize initial human position
        dx = self.np_random.uniform(-1.0, 1.0)
        dy = self.np_random.uniform(-1.0, 1.0)
        bbox_size = self.np_random.uniform(0.1, 0.5)
        obstacle_flag = self.np_random.choice([0.0, 1.0], p=[0.8, 0.2])  # 20% chance of obstacle

        self.state = np.array([dx, dy, bbox_size, obstacle_flag], dtype=np.float32)
        self.steps = 0

        obs = self.state
        info = {}
        return obs, info

    def step(self, action):
        dx, dy, bbox_size, obstacle_flag = self.state

        # Apply action effects (toy dynamics)
        if action == 0:   # forward
            bbox_size += 0.01
        elif action == 1: # back
            bbox_size -= 0.01
        elif action == 2: # left
            dx -= 0.05
        elif action == 3: # right
            dx += 0.05
        elif action == 4: # up
            dy += 0.05
        elif action == 5: # down
            dy -= 0.05
        elif action == 6: # rotate cw
            dx += 0.02
        elif action == 7: # rotate ccw
            dx -= 0.02
        elif action == 8: # hover
            pass

        # Clip values
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        bbox_size = np.clip(bbox_size, 0.0, 1.0)

        self.state = np.array([dx, dy, bbox_size, obstacle_flag], dtype=np.float32)
        self.steps += 1

        # Reward function
        reward = -0.1  # step penalty
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            reward += 10  # human centered
        if bbox_size > 0.7:
            reward += 5   # getting closer
        if bbox_size < 0.05:
            reward -= 10  # lost human
        if obstacle_flag == 1 and bbox_size > 0.6:
            reward -= 20  # too close to obstacle

        # Termination conditions
        terminated = bool(bbox_size < 0.05 or bbox_size >= 1.0)
        truncated = self.steps >= self.max_steps
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        # No visual rendering implemented (stub)
        print(f"State={self.state}")

    def close(self):
        pass
