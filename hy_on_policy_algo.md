# callback.on_training_start(locals(), globals())是在做啥？
Using "Codebase"

Searched for text `on_training_start`, 1 result

Searched for text `on_training_start`, 1 result

Searched for regex `on_training_start|on_rollout_start|on_step|on_rollout_end|on_training_end`, 6 results

Searched for text `callback.on_training_start`, 1 result

Searched for text `BaseCallback`, 11 results

Searched for regex `callback.on_rollout_start|callback.on_step|callback.on_rollout_end|callback.on_training_end`, 4 results

Read [](file:///Users/yanghui/projects/opensource/python/Hybrid-Action-PPO/hy_on_policy_algo.py#200-200), lines 200 to 254

Read [](file:///Users/yanghui/projects/opensource/python/Hybrid-Action-PPO/hy_on_policy_algo.py#200-200), lines 200 to 254

Searched for text `update_locals`, 1 result

Searched codebase for "BaseCallback on_training_start on_step update_locals implementation", 8 results

# `callback.on_training_start(locals(), globals())` 详解

这行代码是在**训练开始时调用回调的初始化钩子**，并将当前的本地变量和全局变量传递给回调对象。

## 1. 代码位置与上下文

### 在 [`hy_on_policy_algo.py:217`](hy_on_policy_algo.py ) 中的使用

```python
def learn(
    self: SelfHyOnPolicyAlgorithm,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 1,
    tb_log_name: str = "OnPolicyAlgorithm",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
) -> SelfHyOnPolicyAlgorithm:
    iteration = 0

    # 1. 设置训练环境和回调
    total_timesteps, callback = self._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    # 2. 🎯 调用回调的训练开始钩子
    callback.on_training_start(locals(), globals())

    assert self.env is not None

    # 3. 开始训练循环
    while self.num_timesteps < total_timesteps:
        continue_training = self.collect_rollouts(...)
        # ...
```

## 2. `on_training_start` 方法详解

### 方法签名

```python
class BaseCallback:
    def on_training_start(
        self, 
        locals_: Dict[str, Any],   # 本地变量字典
        globals_: Dict[str, Any]    # 全局变量字典
    ) -> None:
        """
        训练开始时调用一次
        
        这个方法在训练循环开始前被调用，用于：
        - 初始化回调状态
        - 访问训练参数
        - 设置监控指标
        """
        # 保存变量字典供后续使用
        self.locals = locals_
        self.globals = globals_
        
        # 调用用户自定义的钩子
        self._on_training_start()
```

### 内部实现（Stable-Baselines3）

```python
# stable_baselines3/common/callbacks.py

class BaseCallback:
    def on_training_start(
        self, 
        locals_: Dict[str, Any], 
        globals_: Dict[str, Any]
    ) -> None:
        # 保存变量字典
        self.locals = locals_
        self.globals = globals_
        
        # 调用用户可重写的钩子
        self._on_training_start()
    
    def _on_training_start(self) -> None:
        """
        用户可重写此方法来自定义训练开始时的行为
        """
        pass
```

## 3. `locals()` 和 `globals()` 的作用

### `locals()` - 本地变量字典

`locals()` 返回当前作用域内的所有局部变量：

```python
def learn(self, total_timesteps, callback, ...):
    iteration = 0
    total_timesteps, callback = self._setup_learn(...)
    
    # 此时 locals() 包含：
    print(locals())
    # {
    #     'self': <HyPPO object>,
    #     'total_timesteps': 100000,
    #     'callback': <EvalCallback object>,
    #     'log_interval': 1,
    #     'tb_log_name': 'PPO',
    #     'reset_num_timesteps': True,
    #     'progress_bar': False,
    #     'iteration': 0,
    # }
    
    callback.on_training_start(locals(), globals())
```

**回调可以访问的本地变量**：
- `self`：算法实例（可以访问 `self.policy`, `self.env` 等）
- `total_timesteps`：训练总步数
- `callback`：回调对象本身
- `iteration`：当前迭代次数
- 其他 `learn()` 方法的参数

### `globals()` - 全局变量字典

`globals()` 返回当前模块的全局命名空间：

```python
# globals() 包含：
# {
#     'HyOnPolicyAlgorithm': <class 'HyOnPolicyAlgorithm'>,
#     'HyPPO': <class 'HyPPO'>,
#     'np': <module 'numpy'>,
#     'th': <module 'torch'>,
#     ...
# }
```

**通常不常用**，但提供了访问模块级别变量的能力。

## 4. 为什么要传递这些变量？

### 原因1：回调需要访问训练状态

回调可以通过 `self.locals` 访问算法实例和训练参数：

```python
class CustomCallback(BaseCallback):
    def _on_training_start(self) -> None:
        # 访问算法实例
        model = self.locals['self']
        
        # 打印训练配置
        print(f"训练配置:")
        print(f"  总步数: {self.locals['total_timesteps']}")
        print(f"  学习率: {model.learning_rate}")
        print(f"  批量大小: {model.batch_size}")
        print(f"  环境数量: {model.n_envs}")
        
        # 访问策略网络
        policy = model.policy
        print(f"  策略网络参数量: {sum(p.numel() for p in policy.parameters())}")
```

### 原因2：支持高级回调功能

回调可以在训练开始时执行复杂的初始化：

```python
class TensorBoardImageCallback(BaseCallback):
    def _on_training_start(self) -> None:
        # 获取算法实例
        model = self.locals['self']
        
        # 记录网络结构到 TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(model.tensorboard_log)
        
        # 创建示例输入
        dummy_obs = th.zeros((1, *model.observation_space.shape))
        
        # 记录网络图
        writer.add_graph(model.policy, dummy_obs)
        writer.close()
```

### 原因3：灵活的监控和日志

```python
class DetailedLogCallback(BaseCallback):
    def _on_training_start(self) -> None:
        model = self.locals['self']
        
        # 记录初始状态
        self.logger.record("config/total_timesteps", self.locals['total_timesteps'])
        self.logger.record("config/learning_rate", model.learning_rate)
        self.logger.record("config/n_steps", model.n_steps)
        self.logger.record("config/batch_size", model.batch_size)
        
        # 记录环境信息
        self.logger.record("env/num_envs", model.n_envs)
        self.logger.record("env/observation_shape", model.observation_space.shape)
        self.logger.record("env/action_shape", model.action_space.shape)
        
        self.logger.dump()
```

## 5. 完整的回调生命周期

```python
# 训练流程中的回调调用顺序
model.learn(total_timesteps=100000, callback=callback)

# 1. 初始化回调
callback.init_callback(model)  # 在 _setup_learn 中调用

# 2. 🎯 训练开始（传递 locals 和 globals）
callback.on_training_start(locals(), globals())

# 3. 训练循环
for iteration in range(n_iterations):
    # 3.1 Rollout 开始
    callback.on_rollout_start()
    
    # 3.2 收集数据
    for step in range(n_steps):
        # 3.3 每步调用
        callback.update_locals(locals())  # 更新本地变量
        if not callback.on_step():
            return  # 提前停止
    
    # 3.4 Rollout 结束
    callback.on_rollout_end()
    
    # 3.5 更新策略
    model.train()

# 4. 训练结束
callback.on_training_end()
```

## 6. 实际应用示例

### 示例1：记录训练配置

```python
class ConfigLogCallback(BaseCallback):
    """在训练开始时记录所有配置"""
    
    def _on_training_start(self) -> None:
        # 从 locals 获取算法实例
        model = self.locals['self']
        
        # 记录超参数
        config = {
            'total_timesteps': self.locals['total_timesteps'],
            'learning_rate': model.learning_rate,
            'n_steps': model.n_steps,
            'batch_size': model.batch_size,
            'n_epochs': model.n_epochs,
            'gamma': model.gamma,
            'gae_lambda': model.gae_lambda,
            'clip_range': model.clip_range(1.0),  # 初始裁剪范围
            'ent_coef_con': model.ent_coef_con,
            'ent_coef_disc': model.ent_coef_disc,
            'vf_coef': model.vf_coef,
            'max_grad_norm': model.max_grad_norm,
        }
        
        print("训练配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 保存到文件
        import json
        with open('training_config.json', 'w') as f:
            json.dump(config, f, indent=2)

# 使用
callback = ConfigLogCallback()
model.learn(total_timesteps=100000, callback=callback)
```

### 示例2：验证环境设置

```python
class EnvironmentCheckCallback(BaseCallback):
    """验证环境是否正确设置"""
    
    def _on_training_start(self) -> None:
        model = self.locals['self']
        
        # 检查环境
        print("环境检查:")
        print(f"  环境类型: {type(model.env).__name__}")
        print(f"  环境数量: {model.n_envs}")
        print(f"  观察空间: {model.observation_space}")
        print(f"  动作空间: {model.action_space}")
        
        # 检查是否使用 VecNormalize
        if model._vec_normalize_env is not None:
            print("  ✅ 使用 VecNormalize")
            print(f"    归一化观察: {model._vec_normalize_env.norm_obs}")
            print(f"    归一化奖励: {model._vec_normalize_env.norm_reward}")
        else:
            print("  ⚠️  未使用 VecNormalize")
        
        # 检查动作空间类型
        if isinstance(model.action_space, spaces.Dict):
            print("  ✅ 检测到混合动作空间 (Dict)")
        elif isinstance(model.action_space, spaces.Tuple):
            print("  ✅ 检测到混合动作空间 (Tuple)")
        
        # 验证 buffer 大小
        buffer_size = model.n_steps * model.n_envs
        if buffer_size % model.batch_size != 0:
            print(f"  ⚠️  警告: buffer_size ({buffer_size}) 不能被 batch_size ({model.batch_size}) 整除")
        else:
            print(f"  ✅ Buffer 配置正确")
```

### 示例3：初始化自定义指标

```python
class MetricsTrackerCallback(BaseCallback):
    """跟踪自定义指标"""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
    
    def _on_training_start(self) -> None:
        model = self.locals['self']
        
        print("初始化指标跟踪器")
        print(f"  目标训练步数: {self.locals['total_timesteps']}")
        print(f"  预计迭代次数: {self.locals['total_timesteps'] // (model.n_steps * model.n_envs)}")
        
        # 记录初始策略性能
        print("\n评估初始策略...")
        mean_reward, std_reward = self._evaluate_policy(model, n_episodes=10)
        print(f"  初始平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        
        self.initial_reward = mean_reward
    
    def _evaluate_policy(self, model, n_episodes=10):
        """简单的策略评估"""
        rewards = []
        env = model.get_env()
        
        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
            
            rewards.append(episode_reward)
        
        return np.mean(rewards), np.std(rewards)
```

### 示例4：记录网络权重分布

```python
class WeightDistributionCallback(BaseCallback):
    """记录网络权重的初始分布"""
    
    def _on_training_start(self) -> None:
        model = self.locals['self']
        
        print("记录初始权重分布...")
        
        # 遍历所有网络层
        for name, module in model.policy.named_modules():
            if isinstance(module, (th.nn.Linear, th.nn.Conv2d)):
                weight = module.weight.data
                
                # 记录到 TensorBoard
                self.logger.record(
                    f"weights_init/{name}/mean",
                    weight.mean().item()
                )
                self.logger.record(
                    f"weights_init/{name}/std",
                    weight.std().item()
                )
                self.logger.record(
                    f"weights_init/{name}/norm",
                    weight.norm().item()
                )
                
                print(f"  {name}:")
                print(f"    形状: {weight.shape}")
                print(f"    均值: {weight.mean().item():.4f}")
                print(f"    标准差: {weight.std().item():.4f}")
        
        self.logger.dump()
```

## 7. 与其他回调钩子的配合

### 完整的训练流程示例

```python
class ComprehensiveCallback(BaseCallback):
    """综合使用所有回调钩子"""
    
    def _on_training_start(self) -> None:
        """🎯 训练开始时调用"""
        model = self.locals['self']
        print(f"[训练开始] 总步数: {self.locals['total_timesteps']}")
        print(f"[训练开始] 学习率: {model.learning_rate}")
        self.start_time = time.time()
    
    def _on_rollout_start(self) -> None:
        """每次 rollout 开始时调用"""
        print(f"[Rollout开始] 当前步数: {self.num_timesteps}")
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 访问最新的本地变量
        model = self.locals['self']
        
        if self.n_calls % 1000 == 0:
            print(f"[步骤] {self.num_timesteps} 步")
        
        return True  # 继续训练
    
    def _on_rollout_end(self) -> None:
        """每次 rollout 结束时调用"""
        model = self.locals['self']
        
        if len(model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in model.ep_info_buffer])
            print(f"[Rollout结束] 平均奖励: {mean_reward:.2f}")
    
    def _on_training_end(self) -> None:
        """训练结束时调用"""
        elapsed = time.time() - self.start_time
        print(f"[训练结束] 总耗时: {elapsed:.2f} 秒")
        print(f"[训练结束] 总步数: {self.num_timesteps}")

# 使用
model = HyPPO("MlpPolicy", env, verbose=1)
callback = ComprehensiveCallback()
model.learn(total_timesteps=100000, callback=callback)

# 输出：
# [训练开始] 总步数: 100000
# [训练开始] 学习率: 0.0003
# [Rollout开始] 当前步数: 0
# [步骤] 0 步
# [步骤] 1000 步
# ...
# [Rollout结束] 平均奖励: 150.23
# [训练结束] 总耗时: 123.45 秒
# [训练结束] 总步数: 100000
```

## 8. 调试技巧

### 查看可用的本地变量

```python
class DebugCallback(BaseCallback):
    def _on_training_start(self) -> None:
        print("可用的本地变量:")
        for key, value in self.locals.items():
            print(f"  {key}: {type(value)}")
        
        # 输出：
        # 可用的本地变量:
        #   self: <class 'hy_ppo.HyPPO'>
        #   total_timesteps: <class 'int'>
        #   callback: <class 'stable_baselines3.common.callbacks.CallbackList'>
        #   log_interval: <class 'int'>
        #   tb_log_name: <class 'str'>
        #   reset_num_timesteps: <class 'bool'>
        #   progress_bar: <class 'bool'>
        #   iteration: <class 'int'>
```

### 访问算法的所有属性

```python
class InspectCallback(BaseCallback):
    def _on_training_start(self) -> None:
        model = self.locals['self']
        
        print("算法属性:")
        print(f"  policy: {type(model.policy)}")
        print(f"  env: {type(model.env)}")
        print(f"  rollout_buffer: {type(model.rollout_buffer)}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  device: {model.device}")
        print(f"  n_envs: {model.n_envs}")
        print(f"  observation_space: {model.observation_space}")
        print(f"  action_space: {model.action_space}")
```

## 9. 关键要点总结

### 核心概念

1. **`on_training_start`**：在训练循环开始前调用一次
2. **`locals()`**：传递当前作用域的所有局部变量
3. **`globals()`**：传递模块的全局命名空间
4. **访问算法实例**：通过 `self.locals['self']` 获取

### 常见用途

| 用途 | 示例 |
|------|------|
| 记录配置 | 保存超参数到文件 |
| 验证设置 | 检查环境和 buffer 配置 |
| 初始化指标 | 设置自定义跟踪器 |
| 记录初始状态 | 保存初始权重分布 |
| 评估初始性能 | 测试随机初始化的策略 |

### 访问模式

```python
class MyCallback(BaseCallback):
    def _on_training_start(self) -> None:
        # 访问算法实例
        model = self.locals['self']
        
        # 访问训练参数
        total_timesteps = self.locals['total_timesteps']
        
        # 访问策略网络
        policy = model.policy
        
        # 访问环境
        env = model.env
        
        # 访问 buffer
        buffer = model.rollout_buffer
        
        # 访问日志器
        logger = model.logger
```

### 最佳实践

```python
# ✅ 推荐：在训练开始时记录关键信息
class GoodCallback(BaseCallback):
    def _on_training_start(self) -> None:
        model = self.locals['self']
        print(f"开始训练: {self.locals['total_timesteps']} 步")
        print(f"学习率: {model.learning_rate}")

# ⚠️ 注意：不要在 _on_training_start 中修改训练参数
class BadCallback(BaseCallback):
    def _on_training_start(self) -> None:
        # ❌ 错误：训练已经开始，修改参数可能无效
        self.locals['total_timesteps'] = 200000

# ✅ 正确：应该在创建模型时设置参数
model = HyPPO(..., total_timesteps=200000)
```

`callback.on_training_start(locals(), globals())` 是回调系统的核心入口点，它让回调可以在训练开始时访问完整的训练上下文，从而实现强大的监控、日志和自定义功能！🎯