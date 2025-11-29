# F5-TTS 的 DTM（Distillation Transition Matching）

本文档描述了用于加速 F5-TTS 推理（从 32 步减少到 4-8 步）同时保持音频质量的 DTM 框架的实现。

## 概述

DTM 是一种训练技术，通过在冻结的骨干网络顶部训练轻量级 MLP 头来加速预训练扩散模型。关键思想是：

1. **冻结** 预训练的 DiT 骨干网络（335.8M 参数）
2. **训练** 轻量级 MLP 头（~20M 参数）以预测扩散过程中的大跳跃
3. **减少** 推理步骤从 32 步到 4-8 步，同时保持质量

## 架构

### 组件

1. **冻结的 DiT 骨干网络** (`src/f5_tts/model/backbones/dit.py`)
   - 22 层，16 个头，隐藏层大小 1024
   - 335.8M 参数（冻结）
   - 在每个全局时间步提取高层特征 h_t

2. **可训练的 MLP 头** (`src/f5_tts/model/dtm_head.py`)
   - 输入：骨干网络特征（1024 维）+ 流状态（100 维）
   - 6 层带 AdaLN + FFN
   - 隐藏层维度：512，FFN 扩展：4x
   - 输出：速度场（100 维）
   - ~20M 可训练参数

3. **DTM 包装器** (`src/f5_tts/model/dtm.py`)
   - 结合冻结的骨干网络和可训练的头
   - 实现算法 3（训练）和算法 4（推理）
   - 处理全局和微观时间动态

## 实现细节

### 算法 3：训练

对于每个批次：
1. 采样真实 mel 频谱图 X_T 和噪声 X_0
2. 采样离散时间步 t ∈ {1, ..., T-1}
3. 计算 X_t = (1 - t/T)X_0 + (t/T)X_T
4. 提取冻结的骨干网络特征：h_t = backbone(X_t, t)
5. 采样微观时间 s ∈ [0, 1] 和噪声 Y_noise
6. 计算 Y = X_T - X_0 和 Y_s = (1-s)Y_noise + sY
7. 预测速度：v_pred = head(h_t, Y_s, s)
8. 计算 MSE 损失：||v_pred - (Y - Y_noise)||²

### 算法 4：推理

1. 初始化 X_0 ~ N(0, I)
2. 对于 t = 0 到 T-1：
   - 提取 h_t = backbone(X_t, t)
   - 使用 head(h_t, Y_s, s) 从 s=0 到 s=1 求解 Y 的 ODE
   - 更新：X_{t+1} = X_t + (1/T) * Y_final
3. 返回 X_T

### 关键特性

- **非侵入式设计**：所有 DTM 代码都在单独的文件中，不修改现有的 F5-TTS 代码
- **填充掩码处理**：正确处理 TTS 中的可变长度序列
- **灵活配置**：可配置的全局时间步 T 和 ODE 求解器步骤
- **内存高效**：适配 RTX 4090（24GB）

## 创建的文件

```
src/f5_tts/model/
├── dtm_head.py          # MLP 头架构
├── dtm.py               # DTM 包装器模型
└── __init__.py          # 更新了 DTM 导出

src/f5_tts/configs/
└── DTM_F5TTS_Base.yaml  # DTM 训练配置

src/f5_tts/train/
├── train_dtm.py         # DTM 训练脚本
└── test_dtm.py          # 验证测试
```

## 使用方法

### 1. 先决条件

确保您拥有：
- 预训练的 F5-TTS 检查点（例如，`ckpts/F5TTS_Base/model_last.pt`）
- 准备好的训练数据集（例如，Emilia_ZH_EN）
- 安装了所需的依赖项

### 2. 配置

编辑 `src/f5_tts/configs/DTM_F5TTS_Base.yaml`：

```yaml
ckpts:
  backbone_checkpoint_path: ckpts/F5TTS_Base/model_last.pt  # 设置您的检查点路径

model:
  dtm:
    global_timesteps: 8        # T: 推荐 4-8
    ode_solver_steps: 1        # 1 表示单步 Euler
    ode_solver_method: euler   # euler | midpoint
```