from typing import Dict, List, Type, Union, Tuple
import torch as th
from torch import nn
from stable_baselines3.common.utils import get_device


class HyMlpExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int, # 输入特征的维度
        net_arch: Union[List[int], Dict[str, List[int]]], # 网络架构，可以是一个整数列表或一个包含“pi”和“vf”键的字典
        activation_fn: Type[nn.Module], # 激活函数的类型
        device: Union[th.device, str] = "auto", # 设备类型
    ) -> None:
        '''
        这里是构建一个混合动作空间和动作价值的特征提取器，包含离散动作和连续动作的网络结构和价值网络结构
        这里不是输出一个具体的动作或者价值，有点像输出一个嵌入维度
        '''
        super().__init__()
        device = get_device(device)
        policy_net_con: List[nn.Module] = []
        policy_net_disc: List[nn.Module] = []
        value_net: List[nn.Module] = []
        # 初始化最后一层的维度为输入特征的维度
        last_layer_dim_pi = feature_dim 
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            # 这里针对网络架构是字典类型，提取对应的信息
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            
        else:
            # 这里针对网络架构是列表类型，直接使用相同的层次结构
            pi_layers_dims = vf_layers_dims = net_arch
        for curr_layer_dim in pi_layers_dims:
            # 连续动作的特征提取
            policy_net_con.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net_con.append(activation_fn())
            # 离散动作的特征提取
            policy_net_disc.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net_disc.append(activation_fn())
            # 更新上一层输出的特征维度
            last_layer_dim_pi = curr_layer_dim
        # 构建值网络
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi # 记录动作网络最后一层的维度
        self.latent_dim_vf = last_layer_dim_vf # 记录值网络最后一层的维度
        
        # 构建离散动作、连续动作、值网络
        self.policy_net_con = nn.Sequential(*policy_net_con).to(device)
        self.policy_net_disc = nn.Sequential(*policy_net_disc).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor_disc(features), self.forward_actor_con(features), self.forward_critic(features)

    def forward_actor_disc(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net_disc(features)

    def forward_actor_con(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net_con(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
