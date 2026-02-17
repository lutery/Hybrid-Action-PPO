import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from hy_policies import HyActorCriticPolicy
from hy_buffer import HYRolloutBuffer
from hy_base_class import HyBaseAlgorithm

SelfHyOnPolicyAlgorithm = TypeVar("SelfHyOnPolicyAlgorithm", bound="HyOnPolicyAlgorithm")

class HyOnPolicyAlgorithm(HyBaseAlgorithm):
    rollout_buffer: HYRolloutBuffer
    policy: HyActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[HyActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef_con: float ,
        ent_coef_disc: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps # 每次收集样本的步数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef_con = ent_coef_con
        self.ent_coef_disc = ent_coef_disc
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        if _init_setup_model:
            # 初始化模型
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule() # 这里应该是创建学习率衰减的算法
        self.set_random_seed(self.seed) # 设置随机种子，方便复现

        buffer_cls = HYRolloutBuffer # 创建样本缓存
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # 创建动作策略、价值预测网络
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: HYRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        ''' 
        env: 观察环境，这里看起来是强制设置了向量环境
        callBack: 回掉方法
        rollout_buffer: 缓冲区
        n_rollout_steps: 采集的步数
        '''

        # 在收集时将模型策略设置为验证模式
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0 # 采集的步数统计，就算并行环境，也是一步
        rollout_buffer.reset() # 重制缓冲区
        if self.use_sde:
            # 如果使用了动作连续噪音，则在这里先重制噪音的参数
            self.policy.reset_noise(env.num_envs)

        # 通知回掉开始收集数据
        callback.on_rollout_start()

        # 没有到限制的最大步数，则一直进行数据收集
        while n_steps < n_rollout_steps:
            # 如果使用了动作连续噪音，并且达到了采样频率，则重制噪音参数
            # todo self.sde_sample_freq 的实际值，难道限制最大的游戏步数
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix 重制噪音的参数
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict 将观察转换为tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #actions_disc, actions_con, values, log_prob_disc, log_prob_con
                # 返回预测的离散动作、连续动作、环境价值、离散动作的对数概率，连续动作的对数概率
                actions_disc, actions_con, values, log_probs_disc, log_prob_con = self.policy(obs_tensor)
            actions_disc = actions_disc.cpu().numpy()
            actions_con = actions_con.cpu().numpy()

            # Rescale and perform action
            clipped_actions_disc = actions_disc # 模拟范围裁剪，应该是为了保持命名一致
            clipped_actions_con = np.clip(actions_con, self.action_space_con.low, self.action_space_con.high)# 对连续动作进行动作才见，防止越界
            
            # 根据动作空间类型决定动作格式，将离散动作和连续动作组合起来
            if isinstance(self.action_space, spaces.Dict):
                # Dict 类型动作空间
                clipped_actions = np.concatenate([clipped_actions_disc[:,None], clipped_actions_con], axis=1)
            elif isinstance(self.action_space, spaces.Tuple):
                # Tuple 类型动作空间，创建元组列表
                clipped_actions = [(int(disc), con) for disc, con in zip(clipped_actions_disc.flatten(), clipped_actions_con)]
            else:
                raise TypeError(f"Unsupported action space type: {type(self.action_space)}")
                
            # 处理环境可能返回的不同数量的值 执行动作
            step_result = env.step(clipped_actions)
            if len(step_result) == 5:
                # 新版本 gym API: obs, reward, terminated, truncated, info
                new_obs, rewards, terminated, truncated, infos = step_result
                # 将 terminated 和 truncated 合并为 dones
                dones = np.logical_or(terminated, truncated)
            elif len(step_result) == 4:
                # 旧版本 gym API: obs, reward, done, info
                new_obs, rewards, dones, infos = step_result
            else:
                raise ValueError(f"Unexpected number of values returned by env.step(): {len(step_result)}")
            # 每一个环境执行一步算一步，所以这里统计总步数时要统计每一个环境
            self.num_timesteps += env.num_envs

            # 在每个训练步骤中，将当前作用域的局部变量更新到回调对象中，使回调能够实时访问最新的训练状态。
            # 比如记录本的obs、预测的动作，获得的回报等等
            # 这样做也可以避免在采集、训练流程中嵌入了太多的日志采集的代码，使得代码更加的干净
            callback.update_locals(locals())
            if callback.on_step() is False:
                # todo 这里干嘛要返回,可能是为了能够在训练中通过回调控制流程，比如reward达到了指定的值就让其中断返回
                return False

            # 提取info中的信息，存储到缓冲区
            self._update_info_buffer(infos)
            n_steps += 1

            actions_disc = actions_disc.reshape(-1, 1)

            # 这段代码处理的是 Gym/Gymnasium 环境中的 TimeLimit 截断问题，这是强化学习中一个重要但容易被忽视的细节
            # 而本段代码使用的gym版本不会返回 terminated 和 truncated 两个值，而是直接返回 done，所以需要以下代码作为区分
            # 具体查看md
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ): 
                    # 如果生命周期结束，并且info中包含了终止观察值，则计算该终止观察值的环境价值
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    # 将终止观察值的环境价值加到对应的奖励上
                    rewards[idx] += self.gamma * terminal_value

            # 将当前时间步的数据存储到缓冲区
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_disc,
                actions_con,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs_disc, 
                log_prob_con,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep 当前采集步数达到限制后，计算最后一步的环境价值，此时是没有达到时间限制或者游戏结束，所以要单独领出来计算，主要就是为了PPO的经典计算return和advantage
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
    
        # 计算整个样本的优势和回报
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # 回调通知采样结束
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        raise NotImplementedError

    def learn(
        self: SelfHyOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHyOnPolicyAlgorithm:
        '''
        SelfHyOnPolicyAlgorithm:
        total_timesteps: 训练的总步数
        callback: 训练过程中的回掉函数
        log_interval: 日志打印间隔
        tb_log_name:
        reset_num_timesteps:
        progress_bar:

        '''

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        # 这行代码是在训练开始时调用回调的初始化钩子，并将当前的本地变量和全局变量传递给回调对象。
        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            # 先采集样本
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            # 如果遇到了中断，即在回调中收到训练结束的消息，就跳出循环，结束训练
            if continue_training is False:
                break
            
            # 然后开始训练
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos 打印训练过程中的日志
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        # 回调通知训练结束
        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.value_optimizer", "policy.con_optimizer", "policy.disc_optimizer"]

        return state_dicts, []
