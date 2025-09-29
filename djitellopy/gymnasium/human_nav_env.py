import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HumanNavEnv(gym.Env):
    def __init__(self, grid_size=5, max_steps=30):
        super(HumanNavEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # 0=forward, 1=back, 2=left, 3=right, 4=hover
        self.drone_pos = np.array([0.0, grid_size / 2])
        self.human_pos = np.array([grid_size - 1, grid_size / 2])
        self.current_step = 0
        self.prev_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.random.uniform(0, self.grid_size, 2)
        self.human_pos = np.random.uniform(0, self.grid_size, 2)
        while self._get_distance() < 1.5:  # Ensure some initial separation
            self.human_pos = np.random.uniform(0, self.grid_size, 2)
        self.current_step = 0
        self.prev_distance = self._get_distance()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        move_step = 0.1  # Finer steps
        delta_dist = 0
        
        if action == 0:  # forward
            self.drone_pos[0] += move_step
            delta_dist = self.prev_distance - self._get_distance()
        elif action == 1:  # back
            self.drone_pos[0] -= move_step
            delta_dist = self.prev_distance - self._get_distance()
        elif action == 2:  # left
            self.drone_pos[1] -= move_step
            delta_dist = self.prev_distance - self._get_distance()
        elif action == 3:  # right
            self.drone_pos[1] += move_step
            delta_dist = self.prev_distance - self._get_distance()
        # 4: hover
        
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size - 1)
        
        obs = self._get_obs()
        distance = self._get_distance()
        self.prev_distance = distance
        
        reward = -0.2
        if action == 4:
            reward -= 2
        if delta_dist > 0:
            reward += 15 * delta_dist
        if abs(obs[0]) < 0.6 and abs(obs[1]) < 0.6:
            reward += 30
        if distance < self.grid_size * 0.8:
            reward += 10
        if distance > self.grid_size * 0.8:
            reward -= 30
        if distance < 1.0:
            reward += 50
        
        terminated = distance < 1.0
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_distance(self):
        return np.linalg.norm(self.drone_pos - self.human_pos)

    def _get_obs(self):
        diff = self.human_pos - self.drone_pos
        dx = diff[0] / (self.grid_size / 2)
        dy = diff[1] / (self.grid_size / 2)
        bbox_size = np.clip(2.0 / (self._get_distance() + 0.2), 0, 1)
        return np.array([np.clip(dx, -1, 1), np.clip(dy, -1, 1), bbox_size], dtype=np.float32)

    def render(self):
        print(f"Drone: {self.drone_pos}, Human: {self.human_pos}")