# Hybrid-Action-PPO AI编码助手指南

## 项目概述
基于 Stable-Baselines3 的混合动作空间 PPO 实现，支持同时处理**连续动作和离散动作**的强化学习环境。核心论文：[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1903.01344)

## 核心架构模式

### 1. 混合动作空间处理
项目的关键特性是支持 `spaces.Dict` 和 `spaces.Tuple` 两种混合动作空间格式：

```python
# Dict格式（推荐）
action_space = spaces.Dict({
    'continuous_action': spaces.Box(low=..., high=..., shape=(...), dtype=np.float64),
    'discrete_action': spaces.Discrete(n)
})

# Tuple格式
action_space = spaces.Tuple([
    spaces.Discrete(n),      # 离散动作
    spaces.Box(...)          # 连续动作
])
```

**关键实现点**：
- 在 `hy_base_class.py` 的 `__init__` 中自动检测并拆分动作空间为 `action_space_con` 和 `action_space_disc`
- 在 `collect_rollouts` 中根据动作空间类型组装动作（`hy_on_policy_algo.py:135-142`）
- Buffer存储分别处理连续和离散动作（`hy_buffer.py`）

### 2. 三头网络架构（HyMlpExtractor）
使用分离的网络头来独立处理不同类型的输出（`hyper_layer.py`）：

```python
class HyMlpExtractor:
    policy_net_con   # 连续动作策略网络
    policy_net_disc  # 离散动作策略网络
    value_net        # 价值网络
```

**为什么这样设计**：连续动作和离散动作需要不同的分布（高斯 vs Categorical），分离网络头可以独立优化。

### 3. 三个独立的优化器
在 `HyBaseModel` 和子类中维护三个优化器（`hy_policies.py`）：
```python
self.value_optimizer  # 价值函数优化器
self.disc_optimizer   # 离散动作优化器
self.con_optimizer    # 连续动作优化器
```

**训练时调用**：在 `hy_ppo.py:train()` 中分别更新三个优化器，允许不同的学习率和梯度裁剪策略。

### 4. 环境包装流程
遵循 Stable-Baselines3 的环境包装模式，但需要额外处理混合动作空间：

```
原始环境
  → maybe_make_env (创建环境实例)
  → _patch_env (兼容gym版本差异)
  → Monitor (记录episode统计)
  → DummyVecEnv (向量化接口)
  → VecTransposeImage (如果是图像，转换为channels-first)
```

**关键实现**：`hy_base_class.py:_wrap_env()`

## 代码约定

### 命名规范
- **前缀 `Hy`**：所有自定义类使用 `Hy` 前缀（HyPPO, HyBaseAlgorithm, HyRolloutBuffer等）
- **动作后缀**：连续动作用 `_con`，离散动作用 `_disc`
- **中文注释**：代码中大量使用中文注释（保持这一风格）

### 动作处理约定
在环境的 `step()` 中获取动作：
```python
def step(self, action: np.ndarray):
    discrete_action = action[0]        # 第一个元素是离散动作
    continuous_action = action[1:]     # 其余是连续动作
```

### 熵系数分离
PPO 使用独立的熵系数控制探索：
- `ent_coef_con`：连续动作熵系数
- `ent_coef_disc`：离散动作熵系数

这允许对不同动作类型使用不同的探索策略。

## 关键文件与职责

| 文件 | 职责 |
|------|------|
| `hy_ppo.py` | PPO算法实现，计算策略和价值损失 |
| `hy_on_policy_algo.py` | On-policy算法基类，实现 `collect_rollouts` |
| `hy_base_class.py` | 算法抽象基类，处理环境包装和动作空间检测 |
| `hy_policies.py` | Actor-Critic策略网络定义 |
| `hy_buffer.py` | 专门的Rollout Buffer，存储混合动作 |
| `hyper_layer.py` | 三头网络提取器 |
| `playground.py` | 使用示例（Sliding-v0环境） |

## 开发工作流

### 环境要求
```bash
# 关键依赖
pip install stable-baselines3 gymnasium gym gym_hybrid torch numpy
```

**重要环境变量**：
```python
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
```
此设置避免 protobuf 版本冲突（见 `playground.py:11`）。

### Gym版本兼容性
项目使用 `GymToGym21Wrapper` 处理gym API变化（`playground.py:14-54`）：
- 处理 `reset()` 返回格式（旧版返回obs，新版返回obs+info）
- 处理 `step()` 返回值数量（4个 vs 5个）

**添加新环境时**：必须使用此包装器包裹旧版gym环境。

### 测试/训练命令
```bash
# 运行示例训练
python playground.py
```

### 添加新算法
1. 继承 `HyOnPolicyAlgorithm` 或 `HyBaseAlgorithm`
2. 实现 `train()` 方法
3. 在 `policy_aliases` 中注册策略类型
4. 确保支持 `spaces.Dict` 或 `spaces.Tuple` 动作空间

## 常见陷阱

### 1. 动作空间类型检查
始终检查动作空间类型并相应处理：
```python
if isinstance(self.action_space, spaces.Dict):
    # Dict处理逻辑
elif isinstance(self.action_space, spaces.Tuple):
    # Tuple处理逻辑
else:
    raise TypeError("Unsupported action space")
```

### 2. Buffer大小验证
确保 `n_steps * n_envs > batch_size`，否则会导致优势归一化时出现NaN（见 `hy_ppo.py:84-96`）。

### 3. 梯度裁剪
对三个网络分别进行梯度裁剪：
```python
th.nn.utils.clip_grad_norm_(self.policy.mlp_extractor.policy_net_disc.parameters(), self.max_grad_norm)
th.nn.utils.clip_grad_norm_(self.policy.mlp_extractor.policy_net_con.parameters(), self.max_grad_norm)
th.nn.utils.clip_grad_norm_(self.policy.mlp_extractor.value_net.parameters(), self.max_grad_norm)
```

### 4. 动作裁剪
连续动作必须裁剪到动作空间范围（`hy_on_policy_algo.py:131`）：
```python
clipped_actions_con = np.clip(actions_con, self.action_space_con.low, self.action_space_con.high)
```
离散动作不需要裁剪。

## 与 Stable-Baselines3 的差异

- **动作空间**：SB3只支持单一类型，此项目支持混合
- **优化器**：SB3使用单一优化器，此项目使用三个
- **Buffer**：自定义 `HYRolloutBuffer` 分别存储连续和离散动作
- **策略网络**：使用 `HyMlpExtractor` 替代 SB3 的 `MlpExtractor`

保持与SB3 API兼容性，但内部实现针对混合动作空间优化。
