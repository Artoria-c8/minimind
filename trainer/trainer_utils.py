"""
训练工具函数集合

包含MiniMind模型训练所需的通用工具函数：
- 分布式训练初始化和检测
- 模型参数统计和日志记录
- 学习率调度
- 检查点保存和恢复
- 随机种子设置
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model: torch.nn.Module, config) -> None:
    """
    计算并打印模型参数量统计信息。
    
    对于MoE模型，会分别计算总参数量和激活参数量。
    
    Args:
        model: PyTorch模型实例
        config: 模型配置对象，包含MoE相关配置
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: 
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: 
        Logger(f'Model Params: {total:.2f}M')


def is_main_process() -> bool:
    """
    检查当前进程是否为主进程。
    
    在分布式训练中，只有rank=0的进程是主进程。
    在非分布式训练中，总是返回True。
    
    Returns:
        bool: 如果是主进程返回True，否则返回False
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str) -> None:
    """
    在主进程中打印日志信息。
    
    只有主进程会执行打印，避免分布式训练时重复输出。
    
    Args:
        content: 要打印的日志内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """
    计算余弦退火学习率。
    
    使用cosine annealing策略，学习率从初始值逐渐降低到最小值(初始值的10%)。
    
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率
        
    Returns:
        float: 当前步数对应的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode() -> int:
    """
    初始化分布式训练环境。
    
    检测是否在分布式模式下运行，如果是则初始化进程组并设置CUDA设备。
    
    Returns:
        int: 本地rank（设备编号），非分布式模式返回0
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int) -> None:
    """
    设置随机种子以确保实验可复现。
    
    同时设置Python、NumPy和PyTorch的随机种子，
    并配置cuDNN为确定性模式。
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(
    lm_config, 
    weight: str = 'full_sft', 
    model: Optional[torch.nn.Module] = None, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    epoch: int = 0, 
    step: int = 0, 
    wandb: Optional[Any] = None, 
    save_dir: str = '../checkpoints', 
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    保存或加载训练检查点。
    
    支持两种模式：
    1. 保存模式（model不为None）：保存模型权重、优化器状态和训练进度
    2. 加载模式（model为None）：从检查点恢复训练状态
    
    检查点支持跨GPU数量恢复训练，会自动调整step计数。
    
    Args:
        lm_config: 模型配置对象
        weight: 权重文件名前缀
        model: 要保存的模型，None表示加载模式
        optimizer: 优化器实例
        epoch: 当前训练轮数
        step: 当前训练步数
        wandb: wandb/swanlab日志实例
        save_dir: 检查点保存目录
        **kwargs: 其他需要保存的状态（如scaler、scheduler等）
        
    Returns:
        加载模式下返回检查点数据字典，保存模式返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(
    lm_config, 
    from_weight: str = 'pretrain', 
    tokenizer_path: str = '../model', 
    save_dir: str = '../out', 
    device: str = 'cuda'
) -> tuple:
    """
    初始化模型和分词器。
    
    Args:
        lm_config: 模型配置对象
        from_weight: 要加载的预训练权重名称，'none'表示从头开始训练
        tokenizer_path: 分词器路径
        save_dir: 模型权重目录
        device: 运行设备
        
    Returns:
        tuple: (model, tokenizer) 模型和分词器实例
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    支持跳过指定数量批次的采样器。
    
    用于断点续训时跳过已训练过的批次。
    
    Args:
        sampler: 底层采样器或索引列表
        batch_size: 批次大小
        skip_batches: 需要跳过的批次数量
    """
    
    def __init__(self, sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def save_model_checkpoint(
    model: torch.nn.Module,
    lm_config,
    save_dir: str,
    weight_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    wandb: Optional[Any] = None,
    **kwargs
) -> None:
    """
    保存模型检查点的通用函数。
    
    支持DDP模型和torch.compile包装的模型，自动提取原始模型状态。
    
    Args:
        model: 要保存的模型
        lm_config: 模型配置
        save_dir: 保存目录
        weight_name: 权重文件名前缀
        optimizer: 优化器（可选）
        epoch: 当前epoch
        step: 当前step
        wandb: wandb/swanlab实例（可选）
        **kwargs: 其他需要保存到checkpoint的状态
    """
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight_name}_{lm_config.hidden_size}{moe_suffix}.pth'
    
    # 提取原始模型（处理DDP和torch.compile包装）
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    
    # 保存模型权重（half精度以节省空间）
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp_path)
    
    # 保存完整checkpoint用于断点续训
    if optimizer is not None:
        lm_checkpoint(
            lm_config, 
            weight=weight_name, 
            model=model, 
            optimizer=optimizer, 
            epoch=epoch, 
            step=step, 
            wandb=wandb, 
            save_dir='../checkpoints',
            **kwargs
        )
    
    del state_dict
    torch.cuda.empty_cache()