import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from typing import NamedTuple, Optional, Union,Generator, Union
from stable_baselines3.common.buffers import BaseBuffer


def get_action_dim(action_space: spaces.Space) -> tuple:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape)), 0  # (连续动作维度, 离散动作维度)
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 0, 1  # (连续动作维度, 离散动作维度)
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return 0, int(len(action_space.nvec))  # (连续动作维度, 离散动作维度)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return 0, int(action_space.n)  # (连续动作维度, 离散动作维度)
    elif isinstance(action_space, spaces.Dict):
        return int(np.prod(action_space['continuous_action'].shape)), 1  # (连续动作维度, 离散动作维度)
    elif isinstance(action_space, spaces.Tuple):
        # Tuple 类型动作空间，假设第一个是离散动作，第二个是连续动作
        continuous_dim = 0
        discrete_dim = 0
        
        if isinstance(action_space[0], spaces.Box):
            continuous_dim = int(np.prod(action_space[0].shape))
        elif isinstance(action_space[0], (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            discrete_dim = 1 if isinstance(action_space[0], spaces.Discrete) else int(len(action_space[0].nvec)) if isinstance(action_space[0], spaces.MultiDiscrete) else int(action_space[0].n)
            
        if isinstance(action_space[1], spaces.Box):
            continuous_dim = int(np.prod(action_space[1].shape))
        elif isinstance(action_space[1], (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            discrete_dim = 1 if isinstance(action_space[1], spaces.Discrete) else int(len(action_space[1].nvec)) if isinstance(action_space[1], spaces.MultiDiscrete) else int(action_space[1].n)
            
        return continuous_dim, discrete_dim  # (连续动作维度, 离散动作维度)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


# 用来存储每次采样的数据
class HYRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions_con: th.Tensor
    actions_disc: th.Tensor
    old_values: th.Tensor
    old_log_probs_con: th.Tensor
    old_log_probs_disc: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class HYRolloutBuffer(BaseBuffer):
    observations: np.ndarray
    actions_con: np.ndarray
    actions_disc: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs_con: np.ndarray
    log_probs_disc: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int, 
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space) # 应对不同类型的观测空间，获取观测的形状的帮助方法
        self.action_con_dim, self.action_disc_dim = get_action_dim(action_space) # 获取连续和离散动作的维度
        self.pos = 0 # 当前缓存区的位置指针，用于计算添加数据的位置，就算满了也可以采用余数计算
        self.full = False # 缓存区是否已满的标志，这个主要是为了在采样时确保缓存区已满，采用不同的采样方式
        self.device = get_device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()


    def reset(self) -> None:
        '''
        重置缓存区的缓存
        '''
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions_disc = np.zeros((self.buffer_size, self.n_envs, self.action_disc_dim), dtype=np.float32)
        self.actions_con = np.zeros((self.buffer_size, self.n_envs, self.action_con_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) # 存储计算得到的每一步的回报
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs_disc = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs_con = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) # 存储计算得到的每一步的优势函数
        self.generator_ready = False
        self.pos = 0
        self.full = False
        
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        '''
        计算优势函数和回报

        last_values: th.Tensor 采集结束时的new_obs的价值估计 
        dones: np.ndarray 对应采集结束时执行的动作是否导致游戏结束的标识
        '''

        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # 处理下一步的情况，而这里的下一步对应的是缓冲区中当前step的下一步的状态和价值估计
                next_non_terminal = 1.0 - dones # 如果游戏结束则为0，否则为1
                next_values = last_values # 采集结束时的价值估计
            else:
                # 这里使用的step + 1对应的是缓冲区中当前step的下一步的状态和价值估计
                next_non_terminal = 1.0 - self.episode_starts[step + 1] # 每下一步 如果游戏结束则为0，否则为1
                next_values = self.values[step + 1] # 下一步的价值估计

            # self.rewards[step] + self.gamma * next_values * next_non_terminal :有点类似bellman方程的形式计算Q值，不同的是如果游戏结束则不考虑下一步的价值估计，只为reward
            # 减去 self.values[step] 应该是得到预测的value和实际的TD目标之间的差值，也就是TD误差
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            # 计算GAE优势估计，也就是计算TD误差的加权和（一个序列的优势估计，对连续的时间步的采集有优势，有的时候短时间的损失是为了更大的回报），如果为正数则表示实际回报高于预测价值（选择实际的动作，远离现有的动作预测），负数则表示低于预测价值（反之）
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values # 将减去的价值加回去价值估计，得到类似bellman目标的回报值

    def add(
        self,
        obs: np.ndarray,
        action_disc: np.ndarray,
        action_con: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_probs_disc: th.Tensor,
        log_probs_con: th.Tensor,
        ):
        '''
        将一个时间步的数据添加到缓存区
        观测 obs: np.ndarray
        obs对应的离散动作 action_disc: np.ndarray
        obs对应的连续动作 action_con: np.ndarray
        奖励 reward: np.ndarray
        回合开始标志 episode_start: np.ndarray 如果游戏结束则为True，否则为False，这个应该是用来区分不同回合的
        价值估计 value: th.Tensor
        离散动作的对数概率 log_probs_disc: th.Tensor
        连续动作的对数概率 log_probs_con: th.Tensor
        '''
        self.observations[self.pos] = np.array(obs).copy()
        self.actions_disc[self.pos] = np.array(action_disc).copy()
        self.actions_con[self.pos] = np.array(action_con).copy()
        self.log_probs_disc[self.pos] = log_probs_disc.clone().cpu().numpy()
        self.log_probs_con[self.pos] = log_probs_con.clone().cpu().numpy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def filter(self):
        obs = self.observations[:self.pos]
        return np.mean(obs), np.std(obs)

    def get(self, batch_size: Optional[int] = None) -> Generator[HYRolloutBufferSamples, None, None]:
        '''
        获取指定batch_size的小批量数据生成器，如果batch_size为None则返回整个缓存区的数据
        '''
        assert self.full, ""
        # self.buffer_size * self.n_envs：总的数据量
        # 打乱数据的索引顺序，以便后续随机采样
        # ppo训练并非要排序好的数据，而是要打乱的数据，而之前之所以要排序好，主要是为了计算return 和 advantage
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data # 🔑 关键：准备数据（只在第一次调用时执行）,除非reset
        if not self.generator_ready:
            # 准备好缓冲区的缓存key对应的数据，将其从三维张量重塑为二维张量
            _tensor_names = [
                "observations",
                "actions_con",
                "actions_disc",
                "values",
                "log_probs_con",
                "log_probs_disc",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                # 准备训练数据时，将缓冲区数据从三维张量重塑为二维张量，以便后续的批量采样和训练
                # 具体做法是先交换前两个维度，然后将前两个维度展平为一个维度
                # 这样做的目的是将不同环境和时间步的数据混合在一起，方便后续的随机采样 todo 实际调试时验证一下是否如此
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches 如果没有指定batch_size，则返回整个缓存区的数据
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0 # 起始索引
        while start_idx < self.buffer_size * self.n_envs: # 遍历整个缓存区的数据
            # 采用yield每次返回一个小批量的数据
            # indices[start_idx : start_idx + batch_size]：直接从打乱的索引中获取当前批次的索引
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> HYRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        '''
        根据给定的索引获取对应的小批量数据
        '''
        data = (
            self.observations[batch_inds],
            self.actions_con[batch_inds],
            self.actions_disc[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs_con[batch_inds].flatten(),
            self.log_probs_disc[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        # map(self.to_torch, data): 将数据转换为张量
        # *tuple： 将元组解包为位置参数传递给HYRolloutBufferSamples
        return HYRolloutBufferSamples(*tuple(map(self.to_torch, data)))
