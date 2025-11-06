# AWQ Smooth-Only 使用指南

本文档介绍如何使用 AWQ 的 smooth-only 功能，该功能允许您仅应用 AWQ 的权重平滑（smoothing）操作，而不执行量化，从而保持模型权重在原始精度（如 bf16）下。

## 目录

- [功能概述](#功能概述)
- [工作原理](#工作原理)
- [使用方法](#使用方法)
- [示例](#示例)
- [FAQ](#faq)

## 功能概述

### 什么是 AWQ Smooth-Only？

AWQ (Activation-aware Weight Quantization) 算法包含两个主要步骤：

1. **Smoothing（平滑）**：根据激活值统计调整权重，重新分配量化难度
2. **Quantization（量化）**：将权重量化为低比特（如 int4）

`smooth_only` 模式允许您**仅执行第一步**，保持权重在高精度（bf16/fp16），同时获得 AWQ 的权重调整效果。

### 为什么需要 Smooth-Only？

- ✅ 分析 smoothing 对模型的独立影响
- ✅ 为自定义量化方案准备预处理模型
- ✅ 研究和实验 AWQ 算法的不同组成部分
- ✅ 保持高精度的同时优化权重分布

### 关键特性

| 特性 | 正常 AWQ | Smooth-Only AWQ |
|------|---------|----------------|
| 权重平滑 | ✅ | ✅ |
| 权重量化 | ✅ (int4) | ❌ (保持 bf16) |
| 激活统计收集 | ✅ | ✅ |
| 需要 calibration data | ✅ | ✅ |
| 模型精度 | int4 | bf16/fp16 |
| 数学等价性 | 近似 | 完全等价* |

*在浮点精度范围内

## 工作原理

### AWQ Smoothing 流程

```
                      Calibration Data
                            ↓
                    ┌───────────────────┐
                    │  Forward Passes   │
                    │  (收集激活统计)     │
                    └───────────────────┘
                            ↓
                    ┌───────────────────┐
                    │ 计算 Smooth Scales│
                    │  (per-channel)    │
                    └───────────────────┘
                            ↓
                    ┌───────────────────┐
                    │   应用 Scales     │
                    │ smooth_layer.W /= s │
                    │ balance_layer.W *= s│
                    └───────────────────┘
                            ↓
                  ┌─────────────────────────┐
                  │                         │
       ┌──────────▼──────────┐   ┌─────────▼─────────┐
       │  Normal AWQ Mode    │   │ Smooth-Only Mode  │
       │  继续量化 → int4     │   │ 停止 → bf16       │
       └─────────────────────┘   └───────────────────┘
```

### Smooth-Only 跳过的步骤

在 `smooth_only=True` 模式下，以下操作被跳过：

1. ❌ 量化配置初始化 (`QuantizationMixin.initialize_quantization`)
2. ❌ 量化 observers 注册 (`QuantizationMixin.start_calibration`)
3. ❌ 量化参数计算 (`update_weight_zp_scale`)
4. ❌ QDQ（Quantize-Dequantize）操作启用

### Smooth-Only 保留的操作

以下操作正常执行：

1. ✅ AWQ mappings 解析（确定哪些层需要 smooth）
2. ✅ Activation cache hooks 注册（收集激活统计）
3. ✅ Calibration data forward passes（运行校准数据）
4. ✅ Smooth scales 计算（基于激活和权重统计）
5. ✅ 权重调整应用（应用 scales 到模型权重）

## 使用方法

### 方法 1：Python API（推荐）

```python
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.entrypoints.oneshot import oneshot

# 创建 AWQ modifier with smooth_only=True
modifier = AWQModifier(
    smooth_only=True,        # 关键参数：仅平滑，不量化
    ignore=["lm_head"],      # 忽略的层
    duo_scaling=True,        # 使用 duo scaling（推荐）
    # 可选：如果遇到 OOM 错误
    # offload_device=torch.device("cpu"),
)

# 应用到模型
oneshot(
    model="/path/to/Qwen3-30B-A3B",
    dataset=calibration_data,
    recipe={"one_shot_stage": {"quant_modifiers": {"AWQModifier": modifier}}},
    output_dir="/path/to/smoothed-model",
    num_calibration_samples=512,
)
```

### 方法 2：YAML 配置

创建配置文件 `awq_smooth_only.yaml`：

```yaml
one_shot_stage:
  quant_modifiers:
    AWQModifier:
      smooth_only: true
      ignore:
        - lm_head
      duo_scaling: true
      # offload_device: "cpu"  # 可选
```

使用配置：

```python
from llmcompressor.entrypoints.oneshot import oneshot

oneshot(
    model="/path/to/Qwen3-30B-A3B",
    dataset=calibration_data,
    recipe="awq_smooth_only.yaml",
    output_dir="/path/to/smoothed-model",
    num_calibration_samples=512,
)
```

### 方法 3：命令行脚本

使用提供的示例脚本：

```bash
python examples/awq_smooth_only_example.py \
    --model_path /path/to/Qwen3-30B-A3B \
    --output_path /path/to/smoothed-model \
    --num_samples 512 \
    --calibration_dataset HuggingFaceH4/ultrachat_200k
```

## 示例

### 示例 1：基础使用（Qwen3 30B A3B）

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.entrypoints.oneshot import oneshot

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

# 2. 准备 calibration data
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
dataset = dataset.shuffle(seed=42).select(range(512))

def preprocess(sample):
    text = sample["messages"][0]["content"]
    return tokenizer(text, padding=False, max_length=2048, truncation=True)

calibration_data = [preprocess(s) for s in dataset]

# 3. 创建 smooth-only modifier
modifier = AWQModifier(smooth_only=True, ignore=["lm_head"])

# 4. 应用 smoothing
oneshot(
    model="Qwen/Qwen3-30B-A3B",
    dataset=calibration_data,
    recipe={"one_shot_stage": {"quant_modifiers": {"AWQModifier": modifier}}},
    output_dir="./qwen3-30b-a3b-smoothed",
    max_seq_length=2048,
    num_calibration_samples=512,
)

print("✓ Smoothing complete! Model saved to ./qwen3-30b-a3b-smoothed")
```

### 示例 2：自定义 Mappings

如果您想自定义哪些层进行 smooth：

```python
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

custom_mappings = [
    AWQMapping(
        smooth_layer="re:.*input_layernorm$",
        balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]
    ),
    AWQMapping(
        smooth_layer="re:.*v_proj$",
        balance_layers=["re:.*o_proj$"]
    ),
    # 添加更多 mappings...
]

modifier = AWQModifier(
    smooth_only=True,
    mappings=custom_mappings,
    ignore=["lm_head"]
)
```

### 示例 3：与量化结合

先应用 smooth，再应用自定义量化：

```python
# Step 1: Apply smoothing only
modifier_smooth = AWQModifier(smooth_only=True)
oneshot(
    model="model_path",
    dataset=calibration_data,
    recipe={"one_shot_stage": {"quant_modifiers": {"AWQModifier": modifier_smooth}}},
    output_dir="./smoothed_model",
)

# Step 2: Load smoothed model and apply custom quantization
from llmcompressor.modifiers.quantization import QuantizationModifier

modifier_quant = QuantizationModifier(
    targets=["Linear"],
    scheme="your_custom_scheme",
)

oneshot(
    model="./smoothed_model",
    dataset=calibration_data,
    recipe={"one_shot_stage": {"quant_modifiers": {"QuantizationModifier": modifier_quant}}},
    output_dir="./quantized_model",
)
```

### 示例 4：验证 Smoothing 效果

```python
import torch
from transformers import AutoModelForCausalLM

# 加载原始模型和 smoothed 模型
model_original = AutoModelForCausalLM.from_pretrained("model_path", torch_dtype=torch.bfloat16)
model_smoothed = AutoModelForCausalLM.from_pretrained("./smoothed_model", torch_dtype=torch.bfloat16)

# 比较权重
for name, param_orig in model_original.named_parameters():
    param_smooth = dict(model_smoothed.named_parameters())[name]

    # 检查精度是否保持
    assert param_orig.dtype == param_smooth.dtype == torch.bfloat16

    # 计算权重变化
    diff = (param_orig - param_smooth).abs().mean()
    print(f"{name}: mean difference = {diff:.6f}")

# 验证推理结果（应该在数值精度范围内一致）
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids

with torch.no_grad():
    output_orig = model_original(input_ids).logits
    output_smooth = model_smoothed(input_ids).logits

    # 在浮点精度内应该一致
    assert torch.allclose(output_orig, output_smooth, rtol=1e-3, atol=1e-3)
    print("✓ Outputs match within tolerance!")
```

## FAQ

### Q1: Smooth-only 需要 calibration data 吗？

**A:** 是的！Smoothing 依赖于激活值统计，必须通过 calibration data 的 forward pass 来收集。

### Q2: Smooth-only 会改变模型输出吗？

**A:** 理论上不会。在浮点精度范围内，smoothed 模型应该与原始模型产生相同的输出，因为 smoothing 只是重新分配了计算，数学上等价。

### Q3: Smooth-only 模型的大小和原始模型一样吗？

**A:** 是的。因为权重仍然是 bf16/fp16，模型大小不变。

### Q4: 可以在 smooth-only 后继续量化吗？

**A:** 可以！您可以先运行 smooth-only，保存模型，然后加载并应用任何量化方案。

### Q5: 支持哪些模型架构？

**A:** 支持所有 AWQ 支持的架构，包括：
- Llama, Mistral, Qwen2/3
- Qwen2Moe, Qwen3Moe（MoE 模型）
- Phi3, Gemma2/3, Cohere, DeepseekV3 等

### Q6: Smooth-only 比正常 AWQ 快吗？

**A:** 稍快一些，因为跳过了量化步骤。但主要时间仍然花在 calibration forward passes 上。

### Q7: 可以不提供 quantization scheme 吗？

**A:** 在 smooth-only 模式下，可以不提供 scheme。如果提供了，会被忽略。

### Q8: _num_bits、_symmetric、_group_size 在 smooth-only 下有用吗？

**A:** 这些参数用于 `_compute_best_scale` 中的 pseudo-quantization（用于计算最佳 scale）。Smooth-only 模式使用默认值：4 bits, asymmetric, group_size=128。

## 技术细节

### 代码修改位置

1. **src/llmcompressor/modifiers/awq/base.py**
   - 添加 `smooth_only: bool = False` 参数 (L125)
   - 修改 `validate_awq_after` 跳过量化配置验证 (L160-168)
   - 修改 `on_initialize` 跳过量化初始化 (L224)
   - 修改 `on_start` 跳过 calibration hooks (L244)
   - 修改 `on_end` 跳过量化步骤 (L278-291)

### 测试文件

- **tests/llmcompressor/modifiers/awq/test_smooth_only.py**: 单元测试
- **examples/verify_awq_smooth_only.py**: 验证脚本

### 示例文件

- **examples/awq_smooth_only_qwen3_moe.yaml**: Qwen3 MoE 配置示例
- **examples/awq_smooth_only_example.py**: 完整使用示例

## 参考资料

- [AWQ 论文](https://arxiv.org/pdf/2306.00978)
- [llm-compressor 文档](https://github.com/vllm-project/llm-compressor)

## 贡献

如有问题或建议，请提交 issue：https://github.com/vllm-project/llm-compressor/issues

---

**最后更新**: 2025-11-05
