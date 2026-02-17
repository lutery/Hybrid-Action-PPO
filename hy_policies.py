import collections
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule
import copy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from hyper_layer import HyMlpExtractor

SelfHyBaseModel = TypeVar("SelfHyBaseModel", bound="HyBaseModel")


class HyBaseModel(nn.Module):
    value_optimizer: th.optim.Optimizer
    disc_optimizer: th.optim.Optimizer
    con_optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
            self.features_extractor_kwargs.update(dict(normalized_image=True))

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        """工厂方法，创建特征提取器，可以设置不同的特征提取器类和参数"""
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def extract_features(self, obs: th.Tensor, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        # self.observation_space这里应该是告诉sb3环境的观察的shape，方便给obs进行预处理
        # normalize_images：表示是否归一化的观察
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs) # 使用公共的特征提取，采集公共的观察特征

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls: Type[SelfHyBaseModel], path: str, device: Union[th.device, str] = "auto") -> SelfHyBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def is_vectorized_observation(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
        """
        Check whether or not the observation is vectorized,
        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy)

        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space), self.observation_space
            )
        return vectorized_env

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

class HyBasePolicy(HyBaseModel, ABC):
    features_extractor: BaseFeaturesExtractor # sb3的公共特征提取类
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        '''
        这边的主要作用就是提取动作空间中的离散和连续部分
        '''
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output
        
        # 处理不同类型的动作空间
        if isinstance(self.action_space, spaces.Dict):
            # Dict 类型动作空间
            self.action_space_disc = self.action_space['discrete_action']
            self.action_space_con = self.action_space['continuous_action']
        elif isinstance(self.action_space, spaces.Tuple):
            # Tuple 类型动作空间，假设第一个是离散动作，第二个是连续动作
            self.action_space_disc = self.action_space[0]
            self.action_space_con = self.action_space[1]
        else:
            raise TypeError(f"Unsupported action space type: {type(self.action_space)}. Expected Dict or Tuple.")

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        使用正交初始化方法初始化权重
        
        参数:
            module: 要初始化的模块
            gain: 缩放因子，控制初始权重的大小
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对线性层和卷积层使用正交初始化
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                # 偏置初始化为0
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """"""

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions_disc, actions_con = self._predict(observation, deterministic=deterministic)
        actions_disc = actions_disc.cpu().numpy().reshape((-1, *self.action_space_disc.shape))
        actions_con = actions_con.cpu().numpy().reshape((-1, *self.action_space_con.shape))
        actions_con = np.clip(actions_con, self.action_space_con.low, self.action_space_con.high)
        # Remove batch dimension if needed
        if not vectorized_env:
            actions_disc = actions_disc.squeeze(axis=0)
            actions_con = actions_con.squeeze(axis=0)
        actions = np.concatenate([actions_disc[:,None],actions_con], axis=-1)
        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.action_space_con.low, self.action_space_con.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.action_space_con.low, self.action_space_con.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))



class HyActorCriticPolicy(HyBasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        '''
        在这里是构建了一个连续、离散动作预测、价值动作预测的网络

        net_arch 是用于定义策略网络和价值网络架构的参数，它控制神经网络的层数和每层的神经元数量。
        '''
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        if net_arch is None:
            # 如果没有定义网络架构，使用默认值
            if features_extractor_class == NatureCNN:
                net_arch = [] # todo 为啥如果是 NatureCNN 就用空列表
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        if not share_features_extractor:
            raise "share_features_extractor must be True"
        # 创建公共特征提取器
        self.features_extractor = self.make_features_extractor()
        # 这里应该说特征提取输出的维度
        self.features_dim = self.features_extractor.features_dim
        # todo 这两个属性是干啥的
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            # todo 参数的作用
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # 根据动作空间类型创建对应概率分布对象的工厂函数
        '''
        def make_proba_distribution(
            action_space: spaces.Space,        # 动作空间
            use_sde: bool = False,             # 是否使用gSDE（仅连续动作）
            dist_kwargs: Optional[Dict[str, Any]] = None  # 分布的额外参数
        ) -> Distribution:
            """
            返回与动作空间对应的概率分布对象
    """ 
        '''
        # 离散动作空间 -> CategoricalDistribution
        # 假设 action_space_disc = spaces.Discrete(5)
        '''
        对角高斯分布（Diagonal Gaussian Distribution）：

        假设：各个动作维度相互独立
        协方差矩阵：对角矩阵（非对角元素为0）
        参数：
        均值 μ：每个动作维度的均值
        标准差 σ：每个动作维度的标准差（以对数形式 log_std 存储）
        '''
        self.action_dist_disc = make_proba_distribution(self.action_space_disc, dist_kwargs=None) # 返回: CategoricalDistribution 实例
        
        # 连续动作空间 -> DiagGaussianDistribution
        # 假设 action_space_con = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_dist_con = make_proba_distribution(self.action_space_con, use_sde=use_sde, dist_kwargs=dist_kwargs) # 返回: DiagGaussianDistribution 实例

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        '''
        构建动作、价值嵌入预测网络
        '''
        self.mlp_extractor = HyMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi # 获取动作网络最后一层的嵌入维度

        # 创建动作的分布对象的网络部分 看起来是可以用于动作采样预测
        # # 这是一个 nn.Linear 层，输入维度为 latent_dim_pi，输出维度为离散动作数量
        # self.action_net_disc 实际上是一个 nn.Linear 线性层，由 CategoricalDistribution.proba_distribution_net() 创建
        self.action_net_disc = self.action_dist_disc.proba_distribution_net(latent_dim=latent_dim_pi)
        # 创建连续动作的输出层和对数标准差参
        '''
        proba_distribution_net  参数说明
        """
        创建表示分布的层和参数：
        - 一个输出层产生高斯分布的均值
        - 另一个参数表示标准差（实际存储为对数形式，允许负值）
        
        参数:
            latent_dim: 策略网络最后一层的维度（动作层之前）
            log_std_init: 对数标准差的初始值
        
        返回:
            (mean_actions, log_std): 元组
                - mean_actions: nn.Linear 层，输出动作均值
                - log_std: nn.Parameter，可学习的对数标准差
        """
        '''
        self.action_net_con, self.log_std = self.action_dist_con.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
        # 构建价值网络的输出预测层
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            # 是否使用正交初始化（Orthogonal Initialization） 的开关，用于控制神经网络权重的初始化方式
            #  为不同模块指定不同的 gain 值
            # 不同的GAIN值的意义查看md文档
            module_gains = {
                self.features_extractor: np.sqrt(2), # CNN特征提取器
                self.mlp_extractor: np.sqrt(2),# 三头网络
                self.action_net_con: 0.01,# 连续动作输出层
                self.action_net_disc: 0.01,# 离散动作输出层
                self.value_net: 1,# 价值网络输出层
            }

            # 对每个模块应用正交初始化
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # 将分离不同部分的价值预测网络的参数提取出来，方便后续构建优化器
        value_parameters = [
            self.value_net.parameters(),  # 价值预测网络
            self.mlp_extractor.value_net.parameters(), # 价值嵌入预测网络
            self.features_extractor.parameters() # 公共特征提取网络 todo 为啥公共特征提取要放在这里优化？
        ]
        # t看起来像一次性将所有的动作价值预测参数全部提取出来
        self.value_parameters = [p for group in value_parameters for p in group]

        # 将所有的离散动作价值预测参数全部提取出来展平，方便后续构建优化器
        disc_parameters = [
            self.action_net_disc.parameters(), # 离散动作预测网络
            self.mlp_extractor.policy_net_disc.parameters() #离散动作嵌入预测网络
        ]
        self.disc_parameters = [p for group in disc_parameters for p in group]
        
        # 将所有连续动作价值预测参数全部提取出来展平，方便后续构建优化器
        con_parameters = [
            self.action_net_con.parameters(), # 连续动作预测网络
            [self.log_std],  # 连续动作方差预测网络
            self.mlp_extractor.policy_net_con.parameters() # 连续动作嵌入预测网络
        ]
        self.con_parameters = [p for group in con_parameters for p in (group if isinstance(group, list) else list(group))]
        
        # 构建参数优化器
        self.value_optimizer = self.optimizer_class(self.value_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.disc_optimizer = self.optimizer_class(self.disc_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.con_optimizer = self.optimizer_class(self.con_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        '''
        obs: 观察
        deterministic： todo 这个参数的含义

        return 返回预测的离散动作、连续动作、环境价值、离散动作的对数概率，连续动作的对数概率
        '''
        
        #提取环境的特征
        features = self.extract_features(obs)
        
        # 对提取的特征分别得到离散动作的嵌入、连续动作的嵌入、价值预测的嵌入，每个嵌入都是独立的预测网络，互补相关
        latent_pi_disc, latent_pi_con, latent_vf = self.mlp_extractor(features)
        # 预测价值Q值
        values = self.value_net(latent_vf)
        
        # 创建一个离散动作的分布采样对象
        distribution_disc = self._get_action_dist_from_latent_disc(latent_pi_disc)
        # 负责从离散动作分布中采样动作并计算对数概率
        # deterministic是决定采样的动作是选择最大概率的动作True（验证）还是随机采样False（探索）
        actions_disc = distribution_disc.get_actions(deterministic=deterministic) # action_disc应该返回的就是具体动作的索引
        log_prob_disc = distribution_disc.log_prob(actions_disc) # 计算给定动作的对数概率

        # 创建一个连续动作的分布采样对象（利用均值和方差计算得到）
        distribution_con = self._get_action_dist_from_latent_con(latent_pi_con)
        # 如果是True，则应该是直接返回均值，如果是false,那么可能是加入方差进行采样
        actions_con = distribution_con.get_actions(deterministic=deterministic)
        log_prob_con = distribution_con.log_prob(actions_con)
        
        # 将预测得到的离散动作和连续动作都展平
        actions_disc = actions_disc.reshape((-1, *self.action_space_disc.shape))
        actions_con = actions_con.reshape((-1, *self.action_space_con.shape))
        return actions_disc, actions_con, values, log_prob_disc, log_prob_con

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        return super().extract_features(obs, self.features_extractor)

    def _get_action_dist_from_latent_disc(self, latent_pi: th.Tensor) -> Distribution:
        '''
        param self_pi： 离散动作的嵌入预测
        '''
        # 将离散动作的嵌入向量转换为动作的 logits（对数几率），用于后续创建 Categorical 分布并采样离散动作
        mean_actions = self.action_net_disc(latent_pi)
        # 这行代码是将 logits 转换为 Categorical 分布对象，使得我们可以从该分布中采样动作、计算概率等操作。
        '''
        self.distribution.sample()       # 采样动作
        self.distribution.log_prob(a)    # 计算动作 a 的对数概率
        self.distribution.entropy()      # 计算分布的熵
        self.distribution.probs          # 获取动作概率（自动 softmax）
        self.distribution.logits         # 获取原始 logits
        '''
        return self.action_dist_disc.proba_distribution(action_logits=mean_actions)

    def _get_action_dist_from_latent_con(self, latent_pi: th.Tensor) -> Distribution:
        # 使用连续动作预测的均值头将连续动作的嵌入向量转换为动作的均值，用于后续创建高斯分布并采样连续动作
        mean_actions = self.action_net_con(latent_pi)
        # 对于方差部分判断类型使用不同的方法
        # self.log_std是一个可以学习的方差值
        # 在proba_distribution内部会构造一个和均值一样的矩阵，然后和均值构建一个正太分布
        if isinstance(self.action_dist_con, DiagGaussianDistribution):
            return self.action_dist_con.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist_con, StateDependentNoiseDistribution):
            return self.action_dist_con.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor,th.Tensor]:
        distribution_disc, distribution_con = self.get_distribution(observation)
        return distribution_disc.get_actions(deterministic=deterministic), distribution_con.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions_disc: th.Tensor, actions_con:th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        '''
        todo

        obs: 观察   
        actions_disc: 离散动作
        actions_con: 连续动作

        返回obs对应的预测价值，真实离散动作对应的预测对数概率，真实连续动作对应的对数概率，预测的离散动作的熵，预测的连续动作的熵
        '''
        
        # 提取特征
        features = self.extract_features(obs)
        # 价值嵌入
        latent_vf = self.mlp_extractor.forward_critic(features)
        # 分离特征，防止梯度传播到特征提取器，这样特征提取器只有价值特征才会去更新，避免多个不同维度的特征去更新相同的权重
        # 导致的特征混乱
        detached_f = features.detach()
        # 提取离散动作和连续动作的嵌入
        latent_pi_disc = self.mlp_extractor.forward_actor_disc(detached_f)
        latent_pi_con = self.mlp_extractor.forward_actor_con(detached_f)
        # 根据预测的离散动作嵌入获取一个离散动作的分布采样器（实际上就是将预测的对数概率封装起来）
        distribution_disc = self._get_action_dist_from_latent_disc(latent_pi_disc)
        # 根据实际执行的动作获取该动作对应的对数概率，即用索引获取对应动作的对数概率
        log_prob_disc = distribution_disc.log_prob(actions_disc)
        # 利用动作的对数概率计算熵
        entropy_disc = distribution_disc.entropy()
        
        # 同样的，用连续动作的嵌入计算连续动作的分布采样器（正太分布）
        distribution_con = self._get_action_dist_from_latent_con(latent_pi_con)
        # 根据实际执行的动作获取对应的动作对数概率
        log_prob_con = distribution_con.log_prob(actions_con)
        # 计算对数概率的熵
        entropy_con = distribution_con.entropy()

        # 预测观察价值
        values = self.value_net(latent_vf)
        return values, log_prob_disc, log_prob_con, entropy_disc, entropy_con

    def get_distribution(self, obs: th.Tensor) -> Tuple[Distribution,Distribution]:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi_disc = self.mlp_extractor.forward_actor_disc(features)
        latent_pi_con = self.mlp_extractor.forward_actor_con(features)
        return self._get_action_dist_from_latent_disc(latent_pi_disc), self._get_action_dist_from_latent_con(latent_pi_con)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


    def reset_noise(self, n_envs: int = 1) -> None:
        # 限制重置噪音仅适用于StateDependentNoiseDistribution的动作分布，即在连续动作的使用了gSDE时
        assert isinstance(self.action_dist_con, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        # 采样动作噪音的权重
        self.action_dist_con.sample_weights(self.log_std, batch_size=n_envs) 

class HyActorCriticCnnPolicy(HyActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # todo 这里cnn没有特殊的自定义是如何实现cnn卷积的？
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class HyMultiInputActorCriticPolicy(HyActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        '''
        todo 讲解这里是干嘛的
        但是这里貌似也没干嘛？就是调用了父类的构造函数，也没什么其他的动作
        '''
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

