import time
import argparse
import random
import warnings
import torch
from typing import Tuple, List, Dict, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, PreTrainedModel, PreTrainedTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params

# 只过滤特定的警告，而不是全部
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# 常量定义
WEIGHT_TYPE_PRETRAIN = 'pretrain'
WEIGHT_TYPE_REASON = 'reason'
LORA_NONE = 'None'
MODEL_LOAD_MODE_LOCAL = 'model'

def init_model(args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    初始化模型和分词器
    
    Args:
        args: 命令行参数
        
    Returns:
        模型和分词器的元组
        
    Raises:
        FileNotFoundError: 当模型文件不存在时
        RuntimeError: 当模型加载失败时
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    except Exception as e:
        raise RuntimeError(f"加载分词器失败: {e}")
    
    # 使用更精确的判断逻辑：检查是否为本地torch权重模式
    load_from_local = args.load_from == MODEL_LOAD_MODE_LOCAL or Path(args.load_from).stem == MODEL_LOAD_MODE_LOCAL
    
    if load_from_local:
        try:
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling
            ))
            moe_suffix = '_moe' if args.use_moe else ''
            ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
            
            # 检查权重文件是否存在
            if not Path(ckp).exists():
                raise FileNotFoundError(f"模型权重文件不存在: {ckp}")
            
            # 使用更安全的加载方式
            try:
                # PyTorch 2.0+ 支持 weights_only 参数
                state_dict = torch.load(ckp, map_location=args.device, weights_only=True)
            except TypeError:
                # 旧版本 PyTorch 不支持 weights_only 参数
                state_dict = torch.load(ckp, map_location=args.device)
            
            model.load_state_dict(state_dict, strict=True)
            
            # 加载 LoRA 权重（如果指定）
            if args.lora_weight != LORA_NONE:
                lora_path = f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'
                if not Path(lora_path).exists():
                    raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
                apply_lora(model)
                load_lora(model, lora_path)
        except Exception as e:
            raise RuntimeError(f"加载本地模型失败: {e}")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"从HuggingFace加载模型失败: {e}")
    
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def validate_args(args) -> None:
    """
    验证命令行参数的有效性
    
    Args:
        args: 命令行参数
        
    Raises:
        ValueError: 当参数值无效时
    """
    if not 0 < args.temperature <= 2.0:
        raise ValueError(f"temperature必须在(0, 2.0]范围内，当前值: {args.temperature}")
    
    if not 0 < args.top_p <= 1.0:
        raise ValueError(f"top_p必须在(0, 1.0]范围内，当前值: {args.top_p}")
    
    if args.historys < 0:
        raise ValueError(f"historys不能为负数，当前值: {args.historys}")
    
    if args.historys % 2 != 0:
        print(f"⚠️  警告: historys建议设置为偶数以保持对话完整性，当前值: {args.historys}")
    
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens必须为正数，当前值: {args.max_new_tokens}")
    
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size必须为正数，当前值: {args.hidden_size}")
    
    if args.num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers必须为正数，当前值: {args.num_hidden_layers}")

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    # 验证参数
    try:
        validate_args(args)
    except ValueError as e:
        print(f"❌ 参数验证失败: {e}")
        return 1
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation: List[Dict[str, str]] = []
    
    # 初始化模型
    try:
        model, tokenizer = init_model(args)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return 1
    
    # 获取输入模式（带异常处理）
    while True:
        try:
            input_mode_str = input('[0] 自动测试\n[1] 手动输入\n请选择: ')
            input_mode = int(input_mode_str)
            if input_mode not in [0, 1]:
                print("❌ 请输入0或1")
                continue
            break
        except ValueError:
            print("❌ 输入无效，请输入数字0或1")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 程序已退出")
            return 0
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    try:
        for prompt in prompt_iter:
            # 跳过空输入
            if not prompt or not prompt.strip():
                if input_mode == 1:
                    print("⚠️  输入为空，请重新输入")
                continue
            
            setup_seed(2026)  # or setup_seed(random.randint(0, 2048))
            if input_mode == 0: 
                print(f'💬: {prompt}')
            
            # 保留历史对话
            conversation = conversation[-args.historys:] if args.historys else []
            conversation.append({"role": "user", "content": prompt})

            # 准备输入
            templates: Dict[str, Any] = {
                "conversation": conversation, 
                "tokenize": False, 
                "add_generation_prompt": True
            }
            if args.weight == WEIGHT_TYPE_REASON: 
                templates["enable_thinking"] = True  # 仅Reason模型使用
            
            if args.weight != WEIGHT_TYPE_PRETRAIN:
                inputs = tokenizer.apply_chat_template(**templates)
            else:
                inputs = tokenizer.bos_token + prompt
                
            inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

            # 生成回复
            print('🤖: ', end='')
            st = time.time()
            
            try:
                generated_ids = model.generate(
                    inputs=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens, 
                    do_sample=True, 
                    streamer=streamer,
                    pad_token_id=tokenizer.pad_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=args.top_p, 
                    temperature=args.temperature, 
                    repetition_penalty=1.0
                )
            except Exception as e:
                print(f"\n❌ 生成失败: {e}\n")
                continue
            
            response = tokenizer.decode(
                generated_ids[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            conversation.append({"role": "assistant", "content": response})
            
            # 计算并显示生成速度
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            elapsed_time = time.time() - st
            
            if args.show_speed:
                # 避免除零错误
                speed = gen_tokens / elapsed_time if elapsed_time > 0 else 0
                print(f'\n[Speed]: {speed:.2f} tokens/s\n')
            else:
                print('\n')
                
    except KeyboardInterrupt:
        print("\n\n👋 对话已终止")
        return 0
    except Exception as e:
        print(f"\n\n❌ 运行时错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())