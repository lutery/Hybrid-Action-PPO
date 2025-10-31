# project.md

# Hybrid-Action-PPO 项目规则

## 1. 项目概述

Hybrid-Action-PPO 是一个基于 Stable Baselines3 的强化学习框架，专门用于处理混合动作空间（参数化动作空间）的环境。该项目实现了 PPO（Proximal Policy Optimization）算法的扩展版本，能够同时处理离散和连续动作。

### 核心特性
- 支持混合动作空间（Hybrid Action Space）
- 基于 Stable Baselines3 框架
- 实现了 PPO 算法的扩展版本
- 支持 Dict 和 Tuple 类型的动作空间
- 分别优化离散动作和连续动作的策略

## 2. 项目结构

```
Hybrid-Action-PPO/
├── hy_base_class.py         # 基础算法类
├── hy_on_policy_algo.py     # 在线策略算法基类
├── hy_ppo.py               # PPO 算法实现
├── hy_policies.py          # 策略网络实现
├── hy_buffer.py            # 回放缓冲区实现
├── hyper_layer.py          # 网络层实现
├── playground.py           # 示例代码
└── README.md              # 项目说明文档
```

## 3. 核心组件说明

### 3.1 动作空间处理
项目支持两种混合动作空间类型：
1. **Dict 类型**：包含 'discrete_action' 和 'continuous_action' 键
2. **Tuple 类型**：第一个元素为离散动作，第二个元素为连续动作

### 3.2 网络架构
- **特征提取器**：从观测空间提取特征
- **策略网络**：分为离散动作策略网络和连续动作策略网络
- **价值网络**：用于评估状态价值
- **优化器**：分别为离散策略、连续策略和价值网络设置独立优化器

### 3.3 训练机制
- 使用 PPO 算法进行训练
- 分别计算离散动作和连续动作的损失函数
- 独立优化离散策略、连续策略和价值网络
- 支持 GAE（Generalized Advantage Estimation）优势估计

## 4. 使用规范

### 4.1 环境要求
动作空间必须是以下两种类型之一：
```python
# Dict 类型动作空间
d = {}
d['continuous_action'] = spaces.Box(low=np.array([]), high=np.array([]), shape=(), dtype=np.float64)
d['discrete_action'] = spaces.Discrete()
self.action_space = spaces.Dict(d)

# Tuple 类型动作空间
self.action_space = spaces.Tuple([
    spaces.Discrete(n),  # 离散动作
    spaces.Box(low=low, high=high, shape=shape)  # 连续动作
])
```

### 4.2 动作获取方式
在 step 函数中获取动作：
```python
def step(self, action: np.ndarray):
    discrete_action = action[0]
    continuous_action = action[1:]
```

### 4.3 模型创建和训练
```python
# 创建模型
model = HyPPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef_con=0.01,   # 连续动作熵系数
    ent_coef_disc=0.01,  # 离散动作熵系数
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1
)

# 开始训练
model.learn(total_timesteps=100000, log_interval=10)
```

## 5. 开发规范

### 5.1 代码风格
- 遵循 PEP 8 代码规范
- 使用类型注解
- 保持与 Stable Baselines3 一致的接口设计

### 5.2 类和方法命名
- 类名使用大驼峰命名法（CamelCase）
- 方法名使用小写字母加下划线（snake_case）
- 私有方法以下划线开头

### 5.3 文档规范
- 所有公共方法必须包含文档字符串
- 复杂算法需要添加注释说明
- README.md 需要保持更新

## 6. 扩展指南

### 6.1 添加新策略
1. 继承 `HyBasePolicy` 类
2. 实现 `_predict` 和 `forward` 方法
3. 在 `HyPPO.policy_aliases` 中注册新策略

### 6.2 自定义网络架构
1. 修改 `HyMlpExtractor` 类或创建新的特征提取器
2. 调整 `net_arch` 参数来定义网络结构

### 6.3 环境适配
1. 确保环境的动作空间符合项目要求
2. 实现正确的 step 函数来处理混合动作

## 7. 注意事项

1. 确保动作空间的正确设置，否则会导致运行时错误
2. 连续动作空间必须有有限的上下界
3. 当使用 gSDE（generalized State-Dependent Exploration）时，仅支持连续动作
4. 训练时注意 batch_size 和 n_steps 的设置，避免出现数值问题
5. 对于 Dict 类型观测空间，应使用 MultiInputPolicy 策略
