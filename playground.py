'''
Total reward:  -112.22435834555466
mean reward:  -1.1222435834555466
'''

import gym
import gym_hybrid
from hy_ppo import HyPPO
import os

# 设置环境变量 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 封装一层gym环境，让stable-baselines3可以使用,无法不封装则会出现因为无法准确识别gym版本导致gym step 参数兼容的报错
class GymToGym21Wrapper(gym.Env):
    """
    将gym.Env包装成gymnasium.Env以兼容ptan
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        result = self.env.reset()
        # 处理不同的reset返回格式
        if isinstance(result, tuple):
            obs = result[0]
            info = result[1] if len(result) > 1 else {}
        else:
            obs = result
            info = {}
        return obs, info
    
    def step(self, action):
        result = self.env.step(action)
        # gym-hybrid可能返回5个值: obs, reward, done, truncated, info
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            raise ValueError(f"Unexpected number of return values from env.step: {len(result)}")
        return obs, reward, done, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
# Initialize the environment
# env = gym.make('Sliding-v0', render_mode='human')
env = GymToGym21Wrapper(gym.make('Sliding-v0'))
# env = common.ProcessFrame(gym.make('Sliding-v0', render_mode='human'))
# env = gym.make('Sliding-v0, render_mode='rgb_array', frameskip=4, repeat_action_probability=0.0)
# print("max max_episode_steps:", env.spec.max_episode_steps)
# print("action space:", env.action_space)
# print("observation space:", env.observation_space)
count_frame = 0

# 创建HPPO模型
model = HyPPO(
    policy="MlpPolicy",  # 使用多层感知机策略
    env=env,
    learning_rate=3e-4,
    n_steps=2048,        # 每次更新收集的步数
    batch_size=64,       # 小批量大小
    n_epochs=10,         # 每次更新的训练轮数
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE参数
    clip_range=0.2,      # PPO裁剪范围
    ent_coef_con=0.01,   # 连续动作熵系数
    ent_coef_disc=0.01,  # 离散动作熵系数
    vf_coef=0.5,         # 价值函数损失系数
    max_grad_norm=0.5,   # 梯度裁剪
    verbose=1
    # 移除 tensorboard_log 参数以禁用 tensorboard 日志
)

# 开始训练
model.learn(
    total_timesteps=100000,
    log_interval=10
    # 移除 tb_log_name 参数以禁用 tensorboard 日志
)

# # Reset the environment to get the initial state
# total_reward = 0
# # Run a loop to play the game
# episoid = 100
# for _ in range(episoid):
#     state = env.reset()
#     while True:
#         # Take a random action
#         # env.render()
#         action = env.action_space.sample()

#         # Get the next state, reward, done flag, and info from the environment
#         state, reward, done, trunc, info = env.step(action=action)
#         if reward != 0:
#             total_reward += reward
#             print("action: ", action)   
#             print("reward: ", reward)
#             print("info: ", info)

#         # If done, reset the environment
#         if done or trunc:
#         #     print("info: ", info)
#         #     print("count_frame: ", count_frame)
#             break

# print("Total reward: ", total_reward)
# print("mean reward: ", total_reward / episoid)

# # Close the environment
# env.close()