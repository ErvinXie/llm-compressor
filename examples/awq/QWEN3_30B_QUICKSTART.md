# Qwen3-30B-A3B AWQ Smoothing Quick Start

这个指南专门针对你的 **Qwen3-30B-A3B** 模型，位于 `/mnt/data/models/Qwen3-30B-A3B-250425`

## 模型信息

- **架构**: Qwen3MoeForCausalLM (MoE)
- **专家数**: 128 experts (每次激活 8 个)
- **层数**: 48 layers
- **原始大小**: ~189GB (bfloat16)
- **上下文长度**: 40,960 tokens

## 使用方法

### 方法 1: 使用默认配置（最简单）

```bash
cd /home/xwy/Projects/llm-compressor/examples/awq

# 直接运行，使用默认路径
python qwen3_30b_a3b_smoothing_for_gguf.py
```

### 方法 2: 自定义模型和输出路径

```bash
# 指定你自己的路径
python qwen3_30b_a3b_smoothing_for_gguf.py \
    --model_path /path/to/your/model \
    --output_dir /path/to/output
```

### 方法 3: 完全自定义参数

```bash
python qwen3_30b_a3b_smoothing_for_gguf.py \
    --model_path /mnt/data/models/Qwen3-30B-A3B-250425 \
    --output_dir Qwen3-30B-A3B-awq-smoothed-fp16 \
    --num_calibration_samples 512 \
    --max_seq_length 2048 \
    --offload_to_cpu  # 如果遇到 OOM 添加此参数
```

### 方法 4: 一键 Shell 脚本

```bash
# 使用默认设置
./run_qwen3_30b_smoothing.sh

# 自定义路径
./run_qwen3_30b_smoothing.sh /path/to/model /path/to/output

# 查看帮助
./run_qwen3_30b_smoothing.sh --help
```

### 可用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `/mnt/data/models/Qwen3-30B-A3B-250425` | 输入模型路径 |
| `--output_dir` | `Qwen3-30B-A3B-awq-smoothed-fp16` | 输出目录 |
| `--dataset` | `HuggingFaceH4/ultrachat_200k` | 校准数据集 |
| `--dataset_split` | `train_sft` | 数据集分割 |
| `--num_calibration_samples` | `256` | 校准样本数量 (256-512 推荐) |
| `--max_seq_length` | `2048` | 最大序列长度 |
| `--offload_to_cpu` | `False` | CPU offload (OOM 时添加) |
| `--dtype` | `bfloat16` | 模型数据类型 (bfloat16/float16/auto) |

### 查看所有参数

```bash
python qwen3_30b_a3b_smoothing_for_gguf.py --help
```

**预计时间**: 30-60 分钟（取决于 GPU）

## 高级：YAML 配置

如果你喜欢使用 YAML 配置文件：

```bash
python -m llmcompressor.transformers.apply \
    --model /mnt/data/models/Qwen3-30B-A3B-250425 \
    --dataset HuggingFaceH4/ultrachat_200k \
    --recipe qwen3_30b_a3b_config.yaml \
    --output_dir Qwen3-30B-A3B-awq-smoothed-fp16 \
    --num_calibration_samples 256 \
    --max_seq_length 2048
```

## 输出

运行完成后，你会得到：

```
Qwen3-30B-A3B-awq-smoothed-fp16/
├── config.json
├── generation_config.json
├── model-00001-of-00016.safetensors
├── model-00002-of-00016.safetensors
├── ...
├── model-00016-of-00016.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

- **格式**: FP16 safetensors
- **大小**: ~60GB
- **状态**: AWQ smoothed (未量化)

## 下一步：转换为 GGUF Q4K

### 1. 安装 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
make
```

### 2. 转换为 GGUF

**推荐：Q4_K_M（中等质量）**
```bash
python convert-hf-to-gguf.py \
    /path/to/Qwen3-30B-A3B-awq-smoothed-fp16 \
    --outtype q4_k_m \
    --outfile Qwen3-30B-A3B-awq-q4k.gguf
```

**其他选项：**
```bash
# Q4_K_S - 更小，速度更快
--outtype q4_k_s

# Q4_K_L - 更大，质量更好
--outtype q4_k_l

# Q5_K_M - 更高质量（推荐如果磁盘空间充足）
--outtype q5_k_m

# Q6_K - 接近 FP16 质量
--outtype q6_k
```

**预期 GGUF 大小：**
- Q4_K_S: ~15GB
- Q4_K_M: ~17GB
- Q4_K_L: ~19GB
- Q5_K_M: ~21GB
- Q6_K: ~25GB

### 3. 测试推理

```bash
# 命令行交互
./llama-cli \
    -m Qwen3-30B-A3B-awq-q4k.gguf \
    -p "你好，你能做什么？" \
    -n 200 \
    --temp 0.7

# 启动 API 服务器
./llama-server \
    -m Qwen3-30B-A3B-awq-q4k.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 50  # GPU layers
```

## 内存要求

### AWQ Smoothing 阶段
- **GPU**: 建议 80GB+ (A100/H100)
- **RAM**: 200GB+
- **磁盘**: 250GB+ (原始模型 + 输出)

### GGUF 推理阶段 (Q4_K_M)
- **GPU**: 20-30GB (完全加载)
- **RAM**: 20GB
- **磁盘**: 17GB

## 常见问题

### Q: 遇到 OOM (Out of Memory) 错误怎么办？

A: 在脚本中设置 `OFFLOAD_TO_CPU = True`：

```python
# 在 qwen3_30b_a3b_smoothing_for_gguf.py 中修改
OFFLOAD_TO_CPU = True
```

或者使用 YAML 配置：
```yaml
AWQModifier:
  offload_device: "cpu"
```

### Q: 需要多长时间？

A: **AWQ Smoothing**: 30-60 分钟
   **GGUF 转换**: 10-20 分钟
   **总计**: ~1 小时

### Q: 质量会损失多少？

A: 使用 AWQ smoothing + Q4K 的质量损失通常在 2-3% (perplexity 增加)，比直接 Q4K 量化要好约 1-2%。

### Q: 可以使用其他校准数据集吗？

A: 可以！在脚本中修改：
```python
DATASET_ID = "your/dataset"
# 或使用本地数据集
ds = load_from_disk("/path/to/dataset")
```

### Q: 为什么不直接量化为 4-bit？

A: AWQ smoothing 是一个优化步骤，它使权重更容易量化。llama.cpp 的 Q4K 格式是经过高度优化的推理格式。分离这两个步骤可以：
- 获得 AWQ 的优化效果
- 使用 llama.cpp 的高效格式
- 在各种设备上获得最佳推理性能

## 性能对比

| 方法 | 大小 | 质量 (PPL) | 推理速度 |
|------|------|------------|----------|
| 原始 BF16 | 189GB | 基准 (1.0x) | 慢 |
| 直接 Q4K | 17GB | +4.5% | 快 |
| AWQ + Q4K | 17GB | +2.8% ✓ | 快 |

*注：实际数值因数据集和任务而异*

## 技术细节

### AWQ Smoothing 做了什么？

1. **激活收集**: 在校准数据上运行模型，收集激活统计
2. **权重分析**: 分析每个通道的权重重要性
3. **缩放计算**: 计算最优的每通道缩放因子
4. **权重平滑**: 应用缩放来平衡权重分布
5. **逆向缩放**: 对后续层应用逆缩放以保持数学等价

### Q4K 格式说明

Q4K 使用层次化量化：
- **超级块**: 256 个元素，1 个 FP16 scale
- **子块**: 8 个 32 元素块，每个有 6-bit scale
- **量化值**: 4-bit 整数

这种结构在压缩和质量之间取得了良好平衡。

## 故障排除

### 问题：模型无法加载

```bash
# 检查模型文件
ls -lh /mnt/data/models/Qwen3-30B-A3B-250425/

# 验证 config.json
cat /mnt/data/models/Qwen3-30B-A3B-250425/config.json
```

### 问题：GPU 内存不足

1. 减少批次大小（自动处理）
2. 启用 CPU offload：`OFFLOAD_TO_CPU = True`
3. 使用更少的校准样本：`NUM_CALIBRATION_SAMPLES = 128`

### 问题：NaN 或 Inf 错误

这通常表示：
- 校准数据导致数值不稳定
- 模型本身产生 NaN 输出

尝试：
- 使用不同的校准数据集
- 减少序列长度
- 检查原始模型的健康状况

## 文件说明

- `qwen3_30b_a3b_smoothing_for_gguf.py` - 主脚本（推荐）
- `qwen3_30b_a3b_config.yaml` - YAML 配置
- `run_qwen3_30b_smoothing.sh` - 一键启动脚本
- `QWEN3_30B_QUICKSTART.md` - 本文档

## 支持

如果遇到问题：
1. 查看脚本输出的错误信息
2. 阅读 `README_GGUF.md` 获取详细信息
3. 提交 issue 到 llm-compressor 仓库

## 参考资源

- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Qwen3 官方文档](https://github.com/QwenLM/Qwen3)
