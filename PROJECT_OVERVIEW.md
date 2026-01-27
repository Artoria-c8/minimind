# MiniMind 项目功能介绍

## 📌 项目概述

**MiniMind** 是一个从零开始训练超小型大语言模型（LLM）的开源项目。其核心目标是：

- **极低成本**：仅需约 3 元人民币 + 2 小时即可在单卡 RTX 3090 上训练出具有对话能力的语言模型
- **完全透明**：所有核心算法均使用 PyTorch 原生代码从零实现，不依赖第三方抽象接口
- **教育友好**：适合 LLM 初学者深入理解大模型的内部机制

---

## 🧠 模型架构

模型基于 **Transformer Decoder-Only** 结构（类似 Llama3），主要特点包括：

| 组件 | 说明 |
|------|------|
| **归一化** | 使用 RMSNorm（预标准化方式） |
| **激活函数** | SwiGLU |
| **位置编码** | RoPE（旋转位置嵌入），支持 YaRN 长度外推 |
| **MoE 支持** | 可选的混合专家架构（DeepSeek-V2 风格） |

### 模型参数规模

| 模型名称 | 参数量 | 推理占用 |
|----------|--------|----------|
| MiniMind2-Small | 26M | ~0.5 GB |
| MiniMind2 | 104M | ~1.0 GB |
| MiniMind2-MoE | 145M | ~1.0 GB |

---

## 📚 训练流程

项目覆盖 LLM 训练的**完整生命周期**：

### 1. 预训练 (Pretrain)

```bash
python trainer/train_pretrain.py
```

让模型从大量文本中学习知识（无监督学习），输出权重文件 `pretrain_*.pth`

### 2. 监督微调 (SFT)

```bash
python trainer/train_full_sft.py
```

教会模型按对话模板与人交流，输出权重文件 `full_sft_*.pth`

### 3. LoRA 微调

```bash
python trainer/train_lora.py
```

高效参数微调，支持领域知识注入（如医疗、自我认知），输出权重文件 `lora_xxx_*.pth`

### 4. 知识蒸馏

```bash
python trainer/train_distillation.py
```

从大模型（教师）向小模型（学生）迁移知识（白盒蒸馏）

### 5. 强化学习 (RLHF / RLAIF)

| 算法 | 训练脚本 | 输出权重 |
|------|----------|----------|
| DPO (直接偏好优化) | `trainer/train_dpo.py` | `dpo_*.pth` |
| PPO (近端策略优化) | `trainer/train_ppo.py` | `ppo_actor_*.pth` |
| GRPO (分组相对策略优化) | `trainer/train_grpo.py` | `grpo_*.pth` |
| SPO (单流策略优化) | `trainer/train_spo.py` | `spo_*.pth` |

### 6. 推理模型训练

```bash
python trainer/train_reason.py
```

支持 DeepSeek-R1 风格的思考链（CoT）蒸馏训练，输出权重文件 `reason_*.pth`

---

## 🛠️ 辅助工具

| 功能 | 文件路径 | 说明 |
|------|----------|------|
| **Tokenizer 训练** | `trainer/train_tokenizer.py` | 自定义词表训练（默认 6400 词） |
| **模型转换** | `scripts/convert_model.py` | PyTorch ↔ Transformers 格式互转 |
| **OpenAI API 服务** | `scripts/serve_openai_api.py` | 兼容 OpenAI API 的推理服务 |
| **API 客户端测试** | `scripts/chat_openai_api.py` | 测试 API 服务接口 |
| **Web Demo** | `scripts/web_demo.py` | 基于 Streamlit 的聊天界面 |
| **模型评估** | `eval_llm.py` | 命令行问答测试 |

---

## 📂 项目结构

```
minimind/
├── model/                    # 模型定义
│   ├── model_minimind.py     # MiniMind 模型核心代码
│   ├── model_lora.py         # LoRA 实现
│   ├── tokenizer.json        # 分词器配置
│   └── tokenizer_config.json
├── trainer/                  # 训练脚本
│   ├── train_pretrain.py     # 预训练
│   ├── train_full_sft.py     # 监督微调
│   ├── train_lora.py         # LoRA 微调
│   ├── train_distillation.py # 知识蒸馏
│   ├── train_dpo.py          # DPO 训练
│   ├── train_ppo.py          # PPO 训练
│   ├── train_grpo.py         # GRPO 训练
│   ├── train_spo.py          # SPO 训练
│   ├── train_reason.py       # 推理模型训练
│   ├── train_tokenizer.py    # 分词器训练
│   └── trainer_utils.py      # 训练工具函数
├── dataset/                  # 数据集模块
│   ├── lm_dataset.py         # 数据集加载器
│   └── dataset.md            # 数据集说明
├── scripts/                  # 实用脚本
│   ├── convert_model.py      # 模型格式转换
│   ├── serve_openai_api.py   # API 服务
│   ├── chat_openai_api.py    # API 测试
│   └── web_demo.py           # Web 演示
├── eval_llm.py               # 模型评估
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

---

## 🔧 生态兼容

项目支持主流工具链：

- **llama.cpp** - C++ 高效推理
- **vLLM** - 高吞吐量推理服务
- **ollama** - 本地大模型运行工具
- **MNN** - 端侧 AI 推理引擎
- **Llama-Factory** - 第三方训练框架
- **transformers / trl / peft** - HuggingFace 生态

---

## 📊 数据集

项目提供完整的开源数据集（需单独下载至 `./dataset/` 目录）：

| 数据集 | 大小 | 用途 |
|--------|------|------|
| `pretrain_hq.jsonl` | 1.6GB | 预训练数据 |
| `sft_mini_512.jsonl` | 1.2GB | SFT 快速训练数据 |
| `sft_512.jsonl` | 7.5GB | SFT 完整数据 |
| `sft_1024.jsonl` | 5.6GB | SFT 中等长度数据 |
| `sft_2048.jsonl` | 9GB | SFT 长文本数据 |
| `dpo.jsonl` | 55MB | 偏好对齐数据 |
| `rlaif-mini.jsonl` | 1MB | RLAIF 训练数据 |
| `r1_mix_1024.jsonl` | 340MB | 推理蒸馏数据 |
| `lora_identity.jsonl` | 22.8KB | 自我认知数据 |
| `lora_medical.jsonl` | 34MB | 医疗问答数据 |

数据集下载地址：
- [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

---

## 🚀 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 测试已有模型

```bash
# 下载模型
git clone https://huggingface.co/jingyaogong/MiniMind2

# 命令行问答
python eval_llm.py --load_from ./MiniMind2

# 启动 Web UI
streamlit run scripts/web_demo.py
```

### 从零训练

```bash
# 1. 下载数据集到 ./dataset/

# 2. 预训练
python trainer/train_pretrain.py

# 3. 监督微调
python trainer/train_full_sft.py

# 4. 测试模型
python eval_llm.py --weight full_sft
```

### 多卡训练

```bash
# DDP 多卡训练
torchrun --nproc_per_node N trainer/train_xxx.py
```

---

## 📖 参考资料

- [项目 GitHub](https://github.com/jingyaogong/minimind)
- [模型下载 - HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- [模型下载 - ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47)
- [在线体验 - 推理模型](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning)
- [在线体验 - 常规模型](https://www.modelscope.cn/studios/gongjy/MiniMind)

---

## 📜 许可证

本项目采用 [Apache-2.0 License](LICENSE) 开源协议。
