"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.base_class import maybe_make_env

SelfHyBaseAlgorithm = TypeVar("SelfHyBaseAlgorithm", bound="HyBaseAlgorithm")
class HyBaseAlgorithm(ABC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {}
    policy: BasePolicy
    observation_space: spaces.Space
    action_space: spaces.Space
    n_envs: int
    lr_schedule: Schedule
    _logger: Logger

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],  # 使用多层感知机策略
        env: Union[GymEnv, str, None],  # 环境实例或环境ID  
        learning_rate: Union[float, Schedule], # 学习率或学习率调度函数
        policy_kwargs: Optional[Dict[str, Any]] = None, # 策略网络的额外参数传递
        stats_window_size: int = 100, # todo
        tensorboard_log: Optional[str] = None, # tensorboard日志目录
        verbose: int = 0, # todo 这个应该是日志等级
        device: Union[th.device, str] = "auto", # 运行的设备
        support_multi_env: bool = False, # 是否支持多环境
        monitor_wrapper: bool = True, # 是否使用Monitor包装器
        seed: Optional[int] = None, # 随机种子
        use_sde: bool = False, # todo
        sde_sample_freq: int = -1, # todo
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None, # todo
    ) -> None:
        '''
        再base类中，主要的工作：
        1. 确认参数和环境是否匹配
        2. 创建并包装环境实例
        '''

        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        # self.policy_class 是一个Type策略类

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs # 策略网络的额外参数传递

        self.num_timesteps = 0  # todo
        # Used for updating schedules
        self._total_timesteps = 0 # todo 总经过时间
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None # 本项目没用到
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] 存储上一帧的观察
        self._last_episode_starts = None  # type: Optional[np.ndarray] 存储
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0 # 感觉应该是完成的生命周期的数量
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque] 创建一个存储模型step返回info信息的缓冲区
        self.ep_success_buffer = None  # type: Optional[deque] todo
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env: Optional[VecNormalize] = None
        # Create and wrap the env if needed
        if env is not None:
            # 根据传入的env参数创建环境实例，如果传入的是字符串则创建对应的环境实例
            # 否则估计就直接将传入的环境实例进行包装并返回
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env
            
            # 处理不同类型的动作空间
            # 分离离散动作和连续动作
            if isinstance(self.action_space, spaces.Dict):
                # Dict 类型动作空间
                self.action_space_con = self.action_space['continuous_action']
                self.action_space_disc = self.action_space['discrete_action']
            elif isinstance(self.action_space, spaces.Tuple):
                # Tuple 类型动作空间，假设第一个是离散动作，第二个是连续动作
                self.action_space_disc = self.action_space[0]
                self.action_space_con = self.action_space[1]
            else:
                raise TypeError(f"Unsupported action space type: {type(self.action_space)}. Expected Dict or Tuple.")

            # get VecNormalize object if needed
            # 获取向量化包装器（可能为空，如果没有使用VecNormalize包装器的话）
            '''
            VecNormalize 是一个特殊的环境包装器，用于：

            标准化观察值（归一化到均值0、方差1）
            标准化奖励
            跟踪运行统计信息（均值、方差）
            需要单独保存：训练时收集的统计信息需要在测试/部署时使用，所以需要单独保存和加载

            获取原始观察：在某些情况下需要访问未标准化的原始观察值 todo 查看
            '''
            self._vec_normalize_env = unwrap_vec_normalize(env) # todo 查看哪里使用

            if supported_action_spaces is not None:
                # 这里应该是判断环境的动作空间类型是否在支持的范围内
                # todo 查看实际传递的值是什么
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                # 这里同样是判断是否支持多环境
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            # 这段代码在判断：当观察空间是字典类型（spaces.Dict）时，用户是否错误地使用了 MlpPolicy 或 CnnPolicy。
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

            if self.use_sde and not isinstance(self.action_space_con, spaces.Box):
                # sde是针对连续动作空间添加噪音进行探索时，所添加的噪音是连续的而不是每次都是随机的
                # 这样可以保证探索采样的连续性，保证模型正常的进行训练
                # 而这个功能仅针对连续动作空间有效，如果是离散动作则不能使用这个方法，所以会在这里进行判断
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space_con, spaces.Box):
                # 这里判断连续动作是否有边界，而不是无穷大
                assert np.all(
                    np.isfinite(np.array([self.action_space_con.low, self.action_space_con.high]))
                ), "Continuous action space must have a finite lower and upper bound"

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        '''
        这里将环境包装为一个向量的环境，如果是图片的环境则进行通道转置（HWC -> CHW）

        param env: 环境实例
        param verbose: 日志等级
        param monitor_wrapper: 是否使用Monitor包装器
        return: 包装后的向量化环境实例
        '''
        if not isinstance(env, VecEnv): # 如果不是并行环境，则包装为并行环境
            # Patch to support gym 0.21/0.26 and gymnasium
            # 复Gym不同版本之间的兼容性问题
            '''
            功能:
                统一 Gym 0.21/0.26 和 Gymnasium 的接口差异
                确保环境符合当前SB3期望的API格式
                处理 reset() 返回值格式（旧版本返回obs，新版本返回obs+info）
                处理 step() 返回值格式差异
            '''
            env = _patch_env(env)
            # 判断是否需要包装 Monitor 包装器
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            # 向量化环境包装，使得可以多个环境并行运行
            # 这里传入一个lambda函数，返回env实例，是因为内部认为传入的参数是一个环境创建函数
            # 而不是一个实例
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # 检查 observation_space 是否有嵌套的空间
        '''
        作用: 检查观察空间是否包含不支持的嵌套结构
        检查内容:
        Dict空间中不能再嵌套Dict或Tuple
        Tuple空间中不能再嵌套Dict或Tuple
        如果发现嵌套: 抛出异常

        # ✅ 允许的观察空间
            obs_space = spaces.Dict({
                'image': spaces.Box(0, 255, (64, 64, 3)),
                'vector': spaces.Box(-1, 1, (4,))
            })

            # ❌ 不允许的嵌套（Dict中嵌套Dict）
            obs_space = spaces.Dict({
                'sensors': spaces.Dict({  # 嵌套的Dict
                    'camera': spaces.Box(...),
                    'lidar': spaces.Box(...)
                })
            })
        '''
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            # VecTransposeImage 应该是一个图像预处理包装器，即将图像数据的通道维度进行转置
            # 将图像数据从 (H, W, C) 转换为 (C, H, W)
            # 如果环境的观察是图片则执行这个包装器流程
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
 
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_policy_from_name(self, policy_name: str) -> Type[BasePolicy]:
        '''
        通过策略别名或者实际的策略类名获取策略类
        '''
        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        if isinstance(callback, list):
            callback = CallbackList(callback)

        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        '''
        total_timesteps: todo
        callback: todo
        reset_num_timesteps: 这个有点像是否重置缓冲区的作用
        tf_log_name: todo
        progress_bar: todo
        '''
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            # 本项目没用到，一定是None
            self.action_noise.reset()

        if reset_num_timesteps:
            # 重置标识
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # 如果没重置则统计总经过时间
            # 应该会有传入True和False的情况
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps # todo 记录起始的时间

        if reset_num_timesteps or self._last_obs is None:
            # 这里应该是判断是否是重置或者第一帧
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment] 记录最近的一帧
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool) # 记录开启状态标识
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        self._vec_normalize_env = unwrap_vec_normalize(env)
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self: SelfHyBaseAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHyBaseAlgorithm:
        """
        """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfHyBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfHyBaseAlgorithm:
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]  # pytype: disable=attribute-error
        return model

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)
