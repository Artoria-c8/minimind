"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model: torch.nn.Module, config: Any) -> None:
    """
    计算并打印模型参数量信息
    
    对于MoE模型，会分别计算总参数量、激活参数量等。
    
    Args:
        model: 模型实例
        config: 模型配置对象
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
    判断当前进程是否为主进程
    
    Returns:
        bool: 如果是主进程或非分布式模式返回True，否则返回False
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str) -> None:
    """
    日志输出函数，只在主进程输出
    
    Args:
        content: 要输出的内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """
    计算当前步数的学习率（余弦退火调度）
    
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率
        
    Returns:
        float: 当前步数的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode() -> int:
    """
    初始化分布式训练模式
    
    Returns:
        int: 本地rank，非DDP模式返回0
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int) -> None:
    """
    设置随机种子以确保结果可复现
    
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
    lm_config: Any,
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
    保存或加载模型检查点
    
    保存模式：保存模型权重和训练状态（优化器、epoch、step等）
    加载模式：从检查点恢复训练状态
    
    Args:
        lm_config: 模型配置对象
        weight: 权重名称前缀
        model: 模型实例（None表示加载模式）
        optimizer: 优化器实例
        epoch: 当前epoch
        step: 当前step
        wandb: wandb对象（用于保存run_id）
        save_dir: 保存目录
        **kwargs: 其他要保存的状态
        
    Returns:
        Optional[Dict[str, Any]]: 加载模式时返回检查点数据，保存模式返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # 保存模式
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # 获取wandb run_id
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # 保存其他状态
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
        return None
    else:
        # 加载模式
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
    lm_config: Any,
    from_weight: str = 'pretrain',
    tokenizer_path: str = '../model',
    save_dir: str = '../out',
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    初始化模型和分词器
    
    Args:
        lm_config: 模型配置对象
        from_weight: 权重名称前缀，'none'表示不加载权重
        tokenizer_path: 分词器路径
        save_dir: 权重保存目录
        device: 设备
        
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: 模型和分词器
        
    Raises:
        FileNotFoundError: 当权重文件不存在时
        RuntimeError: 当加载失败时
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise RuntimeError(f"加载分词器失败: {e}")
    
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        try:
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
        except Exception as e:
            raise RuntimeError(f"加载权重失败: {e}")

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    跳过前N个batch的采样器
    
    用于从检查点恢复训练时跳过已经处理过的batch。
    """
    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        """
        Args:
            sampler: 基础采样器
            batch_size: batch大小
            skip_batches: 要跳过的batch数量
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """迭代器：跳过前N个batch，然后正常返回batch"""
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
        # 处理最后一个不完整的batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:
        """返回采样器长度"""
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)