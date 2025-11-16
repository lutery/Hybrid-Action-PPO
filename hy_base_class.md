# policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, spaces.Dict)
## 判断逻辑

这段代码在判断：**当观察空间是字典类型（`spaces.Dict`）时，用户是否错误地使用了 `MlpPolicy` 或 `CnnPolicy`**。

### 具体说明

1. **条件1**：`policy in ["MlpPolicy", "CnnPolicy"]`
   - 检查用户指定的策略名称是否为 `MlpPolicy` 或 `CnnPolicy`

2. **条件2**：`isinstance(self.observation_space, spaces.Dict)`
   - 检查环境的观察空间是否为字典类型

3. **如果两个条件同时满足**：
   - 抛出错误，提示用户必须使用 `MultiInputPolicy`

## 为什么需要这个检查？

### 不同策略的适用场景

| 策略类型 | 适用的观察空间 | 说明 |
|---------|--------------|------|
| **MlpPolicy** | `Box`、`Discrete` 等单一空间 | 使用多层感知机处理简单的向量观察 |
| **CnnPolicy** | `Box`（图像） | 使用卷积神经网络处理图像观察 |
| **MultiInputPolicy** | `Dict`（多种输入） | 可以同时处理多种类型的输入（如图像+向量） |

### 示例场景

````python
# ❌ 错误用法 - 会触发这个检查
observation_space = spaces.Dict({
    'image': spaces.Box(0, 255, (84, 84, 3)),
    'vector': spaces.Box(-np.inf, np.inf, (4,))
})

model = HyPPO(
    policy="MlpPolicy",  # ❌ 错误！Dict观察空间不能用MlpPolicy
    env=env
)
# 报错：You must use `MultiInputPolicy` when working with dict observation space, not MlpPolicy

# ✅ 正确用法
model = HyPPO(
    policy="MultiInputPolicy",  # ✅ 正确！
    env=env
)
````

## 在您的代码中

查看 `hy_ppo.py:18-22` 中的策略别名映射：

````python
policy_aliases: ClassVar[Dict[str, Type[HyBasePolicy]]] = {
    "MlpPolicy": HyActorCriticPolicy,      # 简单向量输入
    "CnnPolicy": HyActorCriticCnnPolicy,   # 图像输入
    "MultiInputPolicy": HyMultiInputActorCriticPolicy,  # 字典输入（多种类型组合）
}
````

三种策略类对应不同的输入处理能力：
- `HyActorCriticPolicy` - 处理单一向量
- `HyActorCriticCnnPolicy` - 处理图像
- `HyMultiInputActorCriticPolicy` - 处理混合输入（字典）

## 总结

这个检查的目的是**防止用户在使用字典观察空间时选择了不兼容的策略类型**，这是一个常见的配置错误，通过提前检查可以给出清晰的错误提示，避免后续难以调试的运行时错误。


# #codebase self._last_original_obs = self._vec_normalize_env.get_original_obs()是什么？
## 1. 代码位置与上下文

### 使用位置（[`hy_base_class.py:389`](hy_base_class.py )）

```python
def _setup_learn(self, ...):
    if reset_num_timesteps or self._last_obs is None:
        assert self.env is not None
        self._last_obs = self.env.reset()  # 获取最新观察
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        
        # 如果使用了 VecNormalize 包装器
        if self._vec_normalize_env is not None:
            # 保存未归一化的原始观察
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
```

### 属性初始化（[`hy_base_class.py:105`](hy_base_class.py )）

```python
class HyBaseAlgorithm:
    def __init__(self, ...):
        self._last_obs = None  # 存储（可能归一化后的）观察
        # When using VecNormalize:
        self._last_original_obs = None  # 存储原始的、未归一化的观察
```

## 2. 什么是 [`VecNormalize`](hy_base_class.py )？

[`VecNormalize`](hy_base_class.py ) 是 Stable-Baselines3 提供的**观察值归一化包装器**，用于自动标准化环境的观察和奖励。

### 核心功能

```python
from stable_baselines3.common.vec_env import VecNormalize

# 包装环境
env = VecNormalize(env)

# VecNormalize 做什么：
# 1. 跟踪观察值的运行统计（均值和方差）
# 2. 将观察值归一化为均值0、方差1
# 3. 可选地归一化奖励
# 4. 保存归一化的统计信息以便后续使用
```

### 工作原理

```python
# 假设环境返回的原始观察
original_obs = np.array([100.0, 200.0, 50.0])

# VecNormalize 内部维护统计量
running_mean = np.array([150.0, 180.0, 60.0])
running_std = np.array([30.0, 40.0, 20.0])

# 归一化后的观察
normalized_obs = (original_obs - running_mean) / (running_std + epsilon)
# = (100-150)/30, (200-180)/40, (50-60)/20
# = [-1.67, 0.5, -0.5]

# env.reset() 或 env.step() 返回的是归一化后的观察
# 但原始观察被内部保存
```

## 3. 为什么需要 [`_last_original_obs`](hy_base_class.py )？

### 场景1：日志记录和可视化

在训练过程中，我们可能想要记录**原始的**观察值，而不是归一化后的值：

```python
# 如果只保存 _last_obs（归一化后）
self._last_obs = [-1.67, 0.5, -0.5]  # 归一化后，不直观

# 保存 _last_original_obs（原始值）
self._last_original_obs = [100.0, 200.0, 50.0]  # 原始值，容易理解

# 在 TensorBoard 中记录
logger.record("observations/position", self._last_original_obs[0])  # 记录100.0
```

### 场景2：渲染和调试

在调试或渲染环境时，显示原始观察更有意义：

```python
# 错误的做法
env.render()  # 环境使用归一化后的观察渲染，可能不正确

# 正确的做法
# 使用原始观察进行渲染
render_obs = self._last_original_obs if self._vec_normalize_env else self._last_obs
env.render(obs=render_obs)
```

### 场景3：模型评估

在评估模型时，可能需要对比训练时的归一化统计：

```python
# 加载训练好的模型
model = HyPPO.load("model.zip", env=eval_env)

# 如果训练时使用了 VecNormalize
# 需要使用相同的归一化统计
if model._vec_normalize_env is not None:
    # 获取原始观察用于验证
    original_obs = model._vec_normalize_env.get_original_obs()
    normalized_obs = model._last_obs
    
    # 验证归一化是否正确
    assert np.allclose(
        (original_obs - mean) / std,
        normalized_obs
    )
```

## 4. [`get_original_obs()`](hy_base_class.py ) 方法详解

### 方法来源

```python
from stable_baselines3.common.vec_env import VecNormalize

# VecNormalize 的方法
class VecNormalize:
    def get_original_obs(self) -> np.ndarray:
        """
        返回最近一次 reset() 或 step() 返回的原始观察
        （即未经归一化的观察）
        """
        return self._obs_buffer  # 内部保存的原始观察缓冲区
```

### 使用示例

```python
# 创建并包装环境
env = gym.make('CartPole-v1')
env = VecNormalize(DummyVecEnv([lambda: env]))

# 重置环境
normalized_obs = env.reset()  # 返回归一化后的观察
original_obs = env.get_original_obs()  # 获取原始观察

print("归一化后:", normalized_obs)  # 例如: [-0.5, 1.2, -0.3, 0.8]
print("原始值:", original_obs)      # 例如: [0.02, 0.5, -0.1, 0.3]
```

## 5. 完整的工作流程

### 训练开始时（[`_setup_learn`](hy_base_class.py )）

```python
def _setup_learn(self, ...):
    if reset_num_timesteps or self._last_obs is None:
        # 1. 重置环境（可能经过 VecNormalize）
        self._last_obs = self.env.reset()
        
        # 2. 如果使用了归一化
        if self._vec_normalize_env is not None:
            # 3. 保存原始观察
            self._last_original_obs = self._vec_normalize_env.get_original_obs()

# 数据流：
# 环境原始输出 → VecNormalize.reset() → 归一化 → self._last_obs
#                                      ↓
#                        内部保存原始观察 → get_original_obs() → self._last_original_obs
```

### 训练过程中

```python
# 在 collect_rollouts 中（伪代码）
while collecting_data:
    # 1. 使用归一化后的观察进行策略预测
    action = policy(self._last_obs)
    
    # 2. 环境执行动作（可能通过 VecNormalize）
    new_obs, reward, done, info = env.step(action)
    # new_obs 已经是归一化后的
    
    # 3. 更新最新观察
    self._last_obs = new_obs
    
    # 4. 如果使用了归一化，同步更新原始观察
    if self._vec_normalize_env is not None:
        self._last_original_obs = self._vec_normalize_env.get_original_obs()
    
    # 5. 记录日志时使用原始观察
    logger.record("env/obs_0", self._last_original_obs[0])
```

## 6. VecNormalize 的内部实现（简化）

```python
class VecNormalize:
    def __init__(self, venv):
        self.venv = venv
        self._obs_buffer = None  # 保存原始观察
        self.obs_rms = RunningMeanStd()  # 观察的运行统计
    
    def reset(self):
        # 1. 获取原始观察
        obs = self.venv.reset()
        self._obs_buffer = obs.copy()  # 保存原始观察
        
        # 2. 归一化
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs
    
    def step(self, actions):
        # 1. 环境执行动作
        obs, rewards, dones, infos = self.venv.step(actions)
        self._obs_buffer = obs.copy()  # 保存原始观察
        
        # 2. 更新统计量
        self.obs_rms.update(obs)
        
        # 3. 归一化
        normalized_obs = self._normalize_obs(obs)
        normalized_rewards = self._normalize_reward(rewards)
        
        return normalized_obs, normalized_rewards, dones, infos
    
    def _normalize_obs(self, obs):
        """归一化观察"""
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
    
    def get_original_obs(self):
        """返回最近保存的原始观察"""
        return self._obs_buffer
```

## 7. 实际应用示例

### 示例1：带归一化的训练

```python
from stable_baselines3.common.vec_env import VecNormalize

# 1. 创建环境并添加归一化
env = gym.make('Pendulum-v1')
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 2. 创建模型
model = HyPPO("MlpPolicy", env)

# 3. 训练过程中
# model._last_obs 包含归一化后的观察（均值0，方差1）
# model._last_original_obs 包含原始观察（真实物理值）

# 4. 训练
model.learn(total_timesteps=10000)

# 5. 保存模型和归一化统计
model.save("ppo_pendulum")
env.save("vec_normalize.pkl")
```

### 示例2：加载并评估

```python
# 1. 加载模型
model = HyPPO.load("ppo_pendulum")

# 2. 创建评估环境（必须使用相同的归一化）
eval_env = gym.make('Pendulum-v1')
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

# 3. 评估时不更新归一化统计
eval_env.training = False
eval_env.norm_reward = False

# 4. 评估
obs = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    
    # 获取原始观察用于显示
    original_obs = eval_env.get_original_obs()
    print(f"原始角度: {original_obs[0][0]:.2f}")
```

### 示例3：日志记录

```python
class CustomCallback(BaseCallback):
    def _on_step(self):
        # 记录归一化后的观察（用于训练）
        self.logger.record("train/normalized_obs_mean", 
                          self.training_env._last_obs.mean())
        
        # 记录原始观察（用于监控真实值）
        if self.model._vec_normalize_env is not None:
            original_obs = self.model._last_original_obs
            self.logger.record("train/original_obs_mean", 
                             original_obs.mean())
        return True

model.learn(total_timesteps=10000, callback=CustomCallback())
```

## 8. 调试技巧

### 检查归一化是否正常工作

```python
# 训练后检查
if model._vec_normalize_env is not None:
    # 1. 检查统计量
    print("观察均值:", model._vec_normalize_env.obs_rms.mean)
    print("观察标准差:", np.sqrt(model._vec_normalize_env.obs_rms.var))
    
    # 2. 检查最新观察
    print("归一化观察:", model._last_obs)
    print("原始观察:", model._last_original_obs)
    
    # 3. 验证归一化公式
    expected = (model._last_original_obs - model._vec_normalize_env.obs_rms.mean) / \
               np.sqrt(model._vec_normalize_env.obs_rms.var + 1e-8)
    print("归一化一致性:", np.allclose(model._last_obs, expected))
```

### 可视化归一化效果

```python
import matplotlib.pyplot as plt

# 收集数据
original_obs_list = []
normalized_obs_list = []

for _ in range(1000):
    action, _ = model.predict(model._last_obs)
    _, _, _, _ = model.env.step(action)
    
    original_obs_list.append(model._last_original_obs[0].copy())
    normalized_obs_list.append(model._last_obs[0].copy())

# 绘图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(original_obs_list, bins=50)
plt.title("原始观察分布")
plt.xlabel("值")

plt.subplot(1, 2, 2)
plt.hist(normalized_obs_list, bins=50)
plt.title("归一化观察分布（应接近N(0,1)）")
plt.xlabel("值")

plt.show()
```

## 9. 关键要点总结

### 核心概念

1. **[`_last_obs`](hy_base_class.py )**：存储最新的观察（如果使用 VecNormalize 则是归一化后的）
2. **[`_last_original_obs`](hy_base_class.py )**：存储未归一化的原始观察（仅当使用 VecNormalize 时）
3. **[`get_original_obs()`](hy_base_class.py )**：VecNormalize 的方法，返回内部保存的原始观察

### 为什么需要保存原始观察？

- ✅ **日志记录**：TensorBoard 中显示真实的物理值
- ✅ **调试和可视化**：渲染环境时使用原始观察
- ✅ **模型验证**：确保归一化统计的正确性
- ✅ **人类可读性**：原始值比归一化值更容易理解

### 数据流图

```
环境原始输出
    ↓
VecNormalize.reset()/step()
    ↓
    ├─→ 内部保存 → _obs_buffer
    │                    ↓
    │          get_original_obs() → _last_original_obs
    │
    └─→ 归一化处理 → normalized_obs → _last_obs
```

### 实践建议

```python
# ✅ 推荐：使用 VecNormalize 进行观察归一化
env = VecNormalize(env, norm_obs=True, norm_reward=True)
model = HyPPO("MlpPolicy", env)

# ✅ 训练后保存归一化统计
env.save("vec_normalize.pkl")

# ✅ 评估时加载相同的归一化统计
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False  # 不更新统计量

# ✅ 记录日志时使用原始观察
if model._vec_normalize_env is not None:
    logger.record("env/original_obs", model._last_original_obs.mean())
```

### 常见错误

```python
# ❌ 错误1：忘记保存归一化统计
model.save("model.zip")  # 只保存模型，没保存归一化统计
# 导致加载后无法正确评估

# ✅ 正确
model.save("model.zip")
model.env.save("vec_normalize.pkl")  # 保存归一化统计

# ❌ 错误2：评估时继续更新统计
eval_env.training = True  # 错误！会继续更新均值/方差
# 导致评估结果不稳定

# ✅ 正确
eval_env.training = False  # 冻结统计量
```

[`_last_original_obs`](hy_base_class.py ) 是使用 VecNormalize 时非常重要的状态保存，它确保我们在使用归一化提高训练效率的同时，仍能访问和监控环境的原始状态！🎯


# 能否讲解一下MaybeCallback
Using "Codebase"

Searched for text `MaybeCallback`, 8 results

Searched for text `BaseCallback`, 7 results

Searched for text `def _init_callback`, 1 result

# [`MaybeCallback`](hy_base_class.py ) 详解

## 1. 类型定义与来源

### 导入位置

```python
# hy_base_class.py:25
from stable_baselines3.common.type_aliases import MaybeCallback

# hy_ppo.py:11
from stable_baselines3.common.type_aliases import MaybeCallback

# hy_on_policy_algo.py:8
from stable_baselines3.common.type_aliases import MaybeCallback
```

### 类型定义（Stable-Baselines3）

```python
from stable_baselines3.common.callbacks import BaseCallback
from typing import Union, List, Callable, Optional

# MaybeCallback 的实际定义
MaybeCallback = Optional[Union[Callable, BaseCallback, List[BaseCallback]]]
```

**解析**：
- [`Optional`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )：可以是 `None`
- [`Union`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )：可以是以下三种类型之一：
  1. [`Callable`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )：可调用对象（函数）
  2. [`BaseCallback`](hy_base_class.py )：回调基类的实例
  3. [`List[BaseCallback]`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )：回调对象列表

## 2. 什么是 Callback（回调）？

Callback 是在训练过程中**特定时间点自动调用的钩子函数**，用于：
- 📊 **记录日志**：保存额外的指标到 TensorBoard
- 💾 **保存检查点**：定期保存模型
- ⏹️ **提前停止**：达到目标性能后停止训练
- 📈 **监控训练**：实时查看训练状态
- 🔄 **自定义逻辑**：在训练循环中插入自定义代码

## 3. [`BaseCallback`](hy_base_class.py ) 基类详解

### 基本结构

```python
from stable_baselines3.common.callbacks import BaseCallback

class BaseCallback:
    """
    回调基类，定义了训练过程中的钩子方法
    """
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.model = None      # 将在 init_callback 时设置
        self.training_env = None
        self.n_calls = 0       # 回调被调用的次数
        self.num_timesteps = 0 # 当前训练的总步数
        self.locals = {}       # 本地变量字典
        self.globals = {}      # 全局变量字典
        
    def init_callback(self, model) -> None:
        """
        在训练开始前初始化回调
        """
        self.model = model
        self.training_env = model.get_env()
        
    def _on_training_start(self) -> None:
        """训练开始时调用一次"""
        pass
    
    def _on_rollout_start(self) -> None:
        """每次 rollout 开始时调用"""
        pass
    
    def _on_step(self) -> bool:
        """
        每个环境步骤后调用
        
        返回:
            bool: 如果返回 False，训练将停止
        """
        return True
    
    def _on_rollout_end(self) -> None:
        """每次 rollout 结束时调用"""
        pass
    
    def _on_training_end(self) -> None:
        """训练结束时调用一次"""
        pass
```

### 回调调用顺序

```python
# 训练流程中的回调调用顺序
model.learn(total_timesteps=10000)

# 1. 初始化回调
callback.init_callback(model)

# 2. 训练开始
callback._on_training_start()

# 3. 训练循环
for iteration in range(n_iterations):
    # 3.1 Rollout 开始
    callback._on_rollout_start()
    
    # 3.2 收集数据
    for step in range(n_steps):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        
        # 每步调用
        if not callback._on_step():
            return  # 提前停止训练
    
    # 3.3 Rollout 结束
    callback._on_rollout_end()
    
    # 3.4 更新策略
    model.train()

# 4. 训练结束
callback._on_training_end()
```

## 4. 三种 [`MaybeCallback`](hy_base_class.py ) 类型详解

### 类型1：`None`（无回调）

```python
# 不使用任何回调
model = HyPPO("MlpPolicy", env)
model.learn(
    total_timesteps=10000,
    callback=None  # 默认值，不使用回调
)

# 等价于
model.learn(total_timesteps=10000)
```

**使用场景**：快速测试，不需要额外功能

### 类型2：[`Callable`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )（简单函数）

```python
# 定义一个简单的回调函数
def simple_callback(locals_: dict, globals_: dict) -> bool:
    """
    简单的回调函数
    
    参数:
        locals_: 本地变量字典，包含当前训练状态
        globals_: 全局变量字典
    
    返回:
        bool: 是否继续训练（True继续，False停止）
    """
    # 获取当前信息
    self_ = locals_['self']
    num_timesteps = self_.num_timesteps
    
    # 每1000步打印一次
    if num_timesteps % 1000 == 0:
        print(f"当前步数: {num_timesteps}")
    
    # 继续训练
    return True

# 使用函数作为回调
model = HyPPO("MlpPolicy", env)
model.learn(
    total_timesteps=10000,
    callback=simple_callback  # 传入函数
)
```

**内部处理**（[`hy_base_class.py:342-343`](hy_base_class.py )）：

```python
def _init_callback(self, callback: MaybeCallback, progress_bar: bool = False) -> BaseCallback:
    if not isinstance(callback, BaseCallback):
        # 如果是 Callable，会被包装成 BaseCallback
        callback = ConvertCallback(callback)
```

**ConvertCallback 的实现**：

```python
class ConvertCallback(BaseCallback):
    """将简单函数转换为 BaseCallback"""
    
    def __init__(self, callback: Callable):
        super().__init__()
        self.callback = callback
    
    def _on_step(self) -> bool:
        # 调用用户提供的函数
        return self.callback(self.locals, self.globals)
```

### 类型3：[`BaseCallback`](hy_base_class.py )（完整回调类）

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    自定义回调类：每隔一定步数保存模型
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _on_training_start(self) -> None:
        """训练开始时打印信息"""
        if self.verbose > 0:
            print(f"开始训练，模型将保存到: {self.save_path}")
    
    def _on_step(self) -> bool:
        """每步检查是否需要保存"""
        # 每 save_freq 步保存一次
        if self.n_calls % self.save_freq == 0:
            # 计算最近100个episode的平均奖励
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                
                if self.verbose > 0:
                    print(f"步数: {self.num_timesteps}, 平均奖励: {mean_reward:.2f}")
                
                # 如果表现更好，保存模型
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_path = f"{self.save_path}/best_model_{self.num_timesteps}"
                    self.model.save(save_path)
                    if self.verbose > 0:
                        print(f"新最佳模型已保存: {save_path}")
        
        return True  # 继续训练
    
    def _on_training_end(self) -> None:
        """训练结束时保存最终模型"""
        final_path = f"{self.save_path}/final_model"
        self.model.save(final_path)
        if self.verbose > 0:
            print(f"最终模型已保存: {final_path}")
            print(f"最佳平均奖励: {self.best_mean_reward:.2f}")

# 使用自定义回调
model = HyPPO("MlpPolicy", env, verbose=1)
callback = CustomCallback(
    save_freq=2048,      # 每2048步检查一次
    save_path="./models/",
    verbose=1
)
model.learn(
    total_timesteps=100000,
    callback=callback  # 传入 BaseCallback 实例
)
```

### 类型4：[`List[BaseCallback]`](/Users/yanghui/anaconda3/lib/python3.12/typing.py )（多个回调）

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 定义多个回调
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="hy_ppo"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_models/",
    log_path="./eval_logs/",
    eval_freq=2000,
    deterministic=True,
    render=False
)

custom_callback = CustomCallback(
    save_freq=2048,
    save_path="./models/"
)

# 使用回调列表
model = HyPPO("MlpPolicy", env)
model.learn(
    total_timesteps=100000,
    callback=[checkpoint_callback, eval_callback, custom_callback]  # 列表
)
```

**内部处理**（[`hy_base_class.py:338-340`](hy_base_class.py )）：

```python
def _init_callback(self, callback: MaybeCallback, progress_bar: bool = False) -> BaseCallback:
    if isinstance(callback, list):
        # 如果是列表，包装成 CallbackList
        callback = CallbackList(callback)
```

**CallbackList 的实现**：

```python
class CallbackList(BaseCallback):
    """按顺序调用多个回调"""
    
    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks
    
    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback._on_training_start()
    
    def _on_step(self) -> bool:
        # 依次调用所有回调
        for callback in self.callbacks:
            if not callback._on_step():
                return False  # 任何一个返回 False，停止训练
        return True
    
    # ... 其他方法类似
```

## 5. 在项目中的实际使用

### 使用位置

```python
# hy_base_class.py:356-368
def _setup_learn(
    self,
    total_timesteps: int,
    callback: MaybeCallback = None,  # 接收 MaybeCallback
    reset_num_timesteps: bool = True,
    tb_log_name: str = "run",
    progress_bar: bool = False,
) -> Tuple[int, BaseCallback]:
    # ... 其他设置 ...
    
    # 初始化并转换回调
    callback = self._init_callback(callback, progress_bar)
    
    return total_timesteps, callback
```

### 转换流程

```python
# hy_base_class.py:329-349
def _init_callback(
    self,
    callback: MaybeCallback,
    progress_bar: bool = False,
) -> BaseCallback:
    """
    将各种类型的回调统一转换为 BaseCallback
    """
    # 1. 如果是列表，包装成 CallbackList
    if isinstance(callback, list):
        callback = CallbackList(callback)
    
    # 2. 如果不是 BaseCallback，包装成 ConvertCallback
    if not isinstance(callback, BaseCallback):
        callback = ConvertCallback(callback)
    
    # 3. 如果启用进度条，添加 ProgressBarCallback
    if progress_bar:
        callback = CallbackList([callback, ProgressBarCallback()])
    
    # 4. 初始化回调
    callback.init_callback(self)
    
    return callback
```

## 6. Stable-Baselines3 内置的常用回调

### 6.1 [`CheckpointCallback`](hy_base_class.py ) - 定期保存模型

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=5000,           # 每5000步保存一次
    save_path="./checkpoints/",
    name_prefix="hy_ppo",     # 文件前缀
    save_replay_buffer=False,
    save_vecnormalize=True
)

model.learn(total_timesteps=100000, callback=checkpoint_callback)

# 生成的文件：
# ./checkpoints/hy_ppo_5000_steps.zip
# ./checkpoints/hy_ppo_10000_steps.zip
# ./checkpoints/hy_ppo_15000_steps.zip
# ...
```

### 6.2 [`EvalCallback`](hy_base_class.py ) - 定期评估模型

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_env = gym.make("Sliding-v0")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_models/",  # 保存最佳模型
    log_path="./eval_logs/",                # 评估日志
    eval_freq=2000,                         # 每2000步评估一次
    n_eval_episodes=10,                     # 每次评估10个episode
    deterministic=True,                     # 使用确定性策略
    render=False
)

model.learn(total_timesteps=100000, callback=eval_callback)

# 效果：
# - 每2000步在 eval_env 上评估10个episode
# - 如果平均奖励更高，自动保存为最佳模型
```

### 6.3 [`StopTrainingOnRewardThreshold`](hy_base_class.py ) - 达到目标奖励停止

```python
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback

# 定义停止条件
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=200.0,  # 平均奖励达到200停止
    verbose=1
)

# 结合 EvalCallback 使用
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,  # 当有新最佳模型时检查
    eval_freq=2000,
    deterministic=True,
    verbose=1
)

model.learn(total_timesteps=1000000, callback=eval_callback)

# 效果：平均奖励达到200时自动停止训练
```

### 6.4 [`ProgressBarCallback`](hy_base_class.py ) - 显示进度条

```python
# 方式1：通过 progress_bar 参数
model.learn(
    total_timesteps=100000,
    progress_bar=True  # 自动添加 ProgressBarCallback
)

# 方式2：显式传入
from stable_baselines3.common.callbacks import ProgressBarCallback

model.learn(
    total_timesteps=100000,
    callback=ProgressBarCallback()
)

# 输出：
# 100%|██████████| 100000/100000 [01:23<00:00, 1198.34it/s]
```

## 7. 实际应用示例

### 示例1：基础训练（无回调）

```python
# playground.py 的简化版本
model = HyPPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)  # 不使用回调
```

### 示例2：使用简单函数回调

```python
def print_episode_reward(locals_, globals_):
    """打印每个episode的奖励"""
    self_ = locals_['self']
    
    # 检查是否有新的episode信息
    if len(self_.ep_info_buffer) > 0:
        latest_ep = self_.ep_info_buffer[-1]
        print(f"Episode {self_._episode_num}: 奖励={latest_ep['r']:.2f}, 长度={latest_ep['l']}")
    
    return True

model.learn(total_timesteps=50000, callback=print_episode_reward)
```

### 示例3：完整的训练流程（多个回调）

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 1. 检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="sliding_hy_ppo"
)

# 2. 评估回调
eval_env = gym.make("Sliding-v0")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_models/",
    log_path="./eval_logs/",
    eval_freq=2000,
    n_eval_episodes=10,
    deterministic=True
)

# 3. 自定义回调：记录额外指标
class MetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        # 记录离散和连续动作的熵
        if self.n_calls % 1000 == 0:
            # 可以访问 self.model 和 self.training_env
            print(f"步数: {self.num_timesteps}")
            print(f"Episode数: {self.model._episode_num}")
        return True

metrics_callback = MetricsCallback()

# 4. 组合所有回调
model = HyPPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(
    total_timesteps=100000,
    callback=[
        checkpoint_callback,
        eval_callback,
        metrics_callback
    ],
    progress_bar=True  # 额外添加进度条
)

# 结果：
# - 每5000步保存检查点
# - 每2000步评估模型
# - 每1000步打印自定义指标
# - 显示训练进度条
```

### 示例4：提前停止训练

```python
class EarlyStoppingCallback(BaseCallback):
    """
    如果最近100个episode的平均奖励超过阈值，提前停止训练
    """
    def __init__(self, reward_threshold: float, patience: int = 10):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.wait = 0
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            if mean_reward >= self.reward_threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"达到目标奖励 {self.reward_threshold}，停止训练")
                    return False  # 停止训练
            else:
                self.wait = 0
        
        return True  # 继续训练

# 使用
early_stopping = EarlyStoppingCallback(
    reward_threshold=200.0,
    patience=10  # 连续10次检查都达到阈值才停止
)

model.learn(
    total_timesteps=1000000,
    callback=early_stopping
)
```

## 8. 访问训练状态

在回调中可以访问的常用属性：

```python
class MyCallback(BaseCallback):
    def _on_step(self) -> bool:
        # 访问模型
        model = self.model
        
        # 访问训练环境
        env = self.training_env
        
        # 访问当前步数
        num_timesteps = self.num_timesteps
        
        # 访问回调被调用的次数
        n_calls = self.n_calls
        
        # 访问 episode 信息缓冲区
        ep_info_buffer = self.model.ep_info_buffer
        
        # 访问日志器
        logger = self.model.logger
        
        # 记录自定义指标到 TensorBoard
        self.logger.record("custom/my_metric", some_value)
        
        # 访问策略网络
        policy = self.model.policy
        
        # 访问优化器
        disc_optimizer = self.model.policy.disc_optimizer
        con_optimizer = self.model.policy.con_optimizer
        value_optimizer = self.model.policy.value_optimizer
        
        return True
```

## 9. 关键要点总结

### [`MaybeCallback`](hy_base_class.py ) 类型

```python
# 类型定义
MaybeCallback = Optional[Union[Callable, BaseCallback, List[BaseCallback]]]

# 可以是：
1. None                    # 不使用回调
2. Callable                # 简单函数
3. BaseCallback            # 完整回调类
4. List[BaseCallback]      # 多个回调
```

### 转换流程

```
用户传入的 callback
    ↓
isinstance(callback, list)?
    Yes → CallbackList(callback)
    No  ↓
isinstance(callback, BaseCallback)?
    Yes → 直接使用
    No  ↓ (Callable or None)
ConvertCallback(callback)
    ↓
progress_bar=True?
    Yes → CallbackList([callback, ProgressBarCallback()])
    No  ↓
init_callback(model)
    ↓
返回统一的 BaseCallback
```

### 回调生命周期

```
1. init_callback(model)      # 初始化，设置 self.model
2. _on_training_start()      # 训练开始
3. for iteration:
       _on_rollout_start()   # Rollout 开始
       for step:
           _on_step()        # 每步（可以返回 False 停止训练）
       _on_rollout_end()     # Rollout 结束
4. _on_training_end()        # 训练结束
```

### 常用场景

| 场景 | 推荐回调 |
|------|---------|
| 定期保存模型 | [`CheckpointCallback`](hy_base_class.py ) |
| 评估性能 | [`EvalCallback`](hy_base_class.py ) |
| 提前停止 | [`StopTrainingOnRewardThreshold`](hy_base_class.py ) |
| 显示进度 | [`ProgressBarCallback`](hy_base_class.py ) |
| 自定义逻辑 | 继承 [`BaseCallback`](hy_base_class.py ) |

### 最佳实践

```python
# ✅ 推荐：组合多个回调
model.learn(
    total_timesteps=100000,
    callback=[
        CheckpointCallback(...),
        EvalCallback(...),
        CustomCallback(...)
    ],
    progress_bar=True
)

# ✅ 简单场景：使用函数
def simple_log(locals_, globals_):
    print(f"Step: {locals_['self'].num_timesteps}")
    return True

model.learn(total_timesteps=10000, callback=simple_log)

# ✅ 无需回调：直接训练
model.learn(total_timesteps=10000)
```

[`MaybeCallback`](hy_base_class.py ) 是 Stable-Baselines3 中非常灵活的设计，通过统一的接口支持从简单的函数到复杂的回调类，极大地增强了训练过程的可定制性！🎯