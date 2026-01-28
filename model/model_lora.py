"""
LoRA (Low-Rank Adaptation) 模型实现
用于高效的参数高效微调
"""
import torch
from torch import nn
from typing import Optional


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 模块
    
    通过低秩分解的方式对预训练模型进行微调，只需要更新少量参数。
    核心思想是在权重矩阵中引入低秩分解：W + ΔW = W + BA，其中B和A是低秩矩阵。
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA的秩（rank），控制低秩矩阵的大小
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        return self.B(self.A(x))


def apply_lora(model: nn.Module, rank: int = 8) -> None:
    """
    为模型的所有线性层应用LoRA适配器
    
    只对方阵（输入输出维度相同）的线性层应用LoRA，避免维度不匹配的问题。
    
    Args:
        model: 要应用LoRA的模型
        rank: LoRA的秩，默认为8
    """
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 使用闭包正确绑定变量，避免循环引用问题
            def make_forward_fn(orig_fn, lora_layer):
                def forward_with_lora(x):
                    return orig_fn(x) + lora_layer(x)
                return forward_with_lora

            module.forward = make_forward_fn(original_forward, lora)


def load_lora(model: nn.Module, path: str) -> None:
    """
    加载LoRA权重到模型
    
    Args:
        model: 要加载LoRA权重的模型
        path: LoRA权重文件路径
        
    Raises:
        FileNotFoundError: 当权重文件不存在时
        RuntimeError: 当加载失败时
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA权重文件不存在: {path}")
    
    try:
        device = next(model.parameters()).device
        state_dict = torch.load(path, map_location=device)
        # 移除DDP前缀（如果有）
        state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                lora_state = {
                    k.replace(f'{name}.lora.', ''): v 
                    for k, v in state_dict.items() 
                    if f'{name}.lora.' in k
                }
                if lora_state:
                    module.lora.load_state_dict(lora_state, strict=False)
    except Exception as e:
        raise RuntimeError(f"加载LoRA权重失败: {e}")


def save_lora(model: nn.Module, path: str) -> None:
    """
    保存模型的LoRA权重
    
    Args:
        model: 要保存LoRA权重的模型
        path: 保存路径
    """
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f'{clean_name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
