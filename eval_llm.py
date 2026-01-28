import time
import argparse
import random
import warnings
import os
import sys
from typing import Tuple, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')


def init_model(args) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    初始化模型和tokenizer
    
    Args:
        args: 命令行参数对象
        
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: 返回初始化好的模型和tokenizer
        
    Raises:
        FileNotFoundError: 当模型权重文件不存在时
        RuntimeError: 当模型加载失败时
    """
    # 检查tokenizer路径是否存在
    if not os.path.exists(args.load_from):
        raise FileNotFoundError(f"模型加载路径不存在: {args.load_from}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    except Exception as e:
        raise RuntimeError(f"加载tokenizer失败: {str(e)}")
    
    if 'model' in args.load_from:
        # 使用原生torch权重
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 检查权重文件是否存在
        if not os.path.exists(ckp):
            raise FileNotFoundError(f"模型权重文件不存在: {ckp}")
        
        try:
            model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        except Exception as e:
            raise RuntimeError(f"加载模型权重失败: {str(e)}")
        
        # 加载LoRA权重（如果指定）
        if args.lora_weight != 'None':
            lora_path = f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
            try:
                apply_lora(model)
                load_lora(model, lora_path)
            except Exception as e:
                raise RuntimeError(f"加载LoRA权重失败: {str(e)}")
    else:
        # 使用transformers格式权重
        try:
            model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"加载transformers模型失败: {str(e)}")
    
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def validate_args(args) -> None:
    """
    验证命令行参数的有效性
    
    Args:
        args: 命令行参数对象
        
    Raises:
        ValueError: 当参数不合法时
    """
    # 验证temperature范围
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError(f"temperature必须在[0.0, 2.0]范围内，当前值: {args.temperature}")
    
    # 验证top_p范围
    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError(f"top_p必须在[0.0, 1.0]范围内，当前值: {args.top_p}")
    
    # 验证historys必须是偶数
    if args.historys < 0:
        raise ValueError(f"historys必须是非负数，当前值: {args.historys}")
    if args.historys % 2 != 0:
        raise ValueError(f"historys必须是偶数（一问一答），当前值: {args.historys}")
    
    # 验证max_new_tokens
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens必须大于0，当前值: {args.max_new_tokens}")
    
    # 验证hidden_size
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size必须大于0，当前值: {args.hidden_size}")
    
    # 验证num_hidden_layers
    if args.num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers必须大于0，当前值: {args.num_hidden_layers}")


def main():
    """主函数：处理模型推理和对话交互"""
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
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-2，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, choices=[0, 1], help="是否显示decode速度（0=否，1=是）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    # 验证参数
    try:
        validate_args(args)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        sys.exit(1)
    
    # 预定义的测试prompt
    prompts: List[str] = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 初始化模型
    try:
        model, tokenizer = init_model(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ 模型初始化失败: {e}")
        sys.exit(1)
    
    # 获取用户输入模式（带验证）
    while True:
        try:
            input_mode_str = input('[0] 自动测试\n[1] 手动输入\n请选择模式: ')
            input_mode = int(input_mode_str)
            if input_mode in [0, 1]:
                break
            else:
                print("❌ 请输入0或1")
        except ValueError:
            print("❌ 请输入有效的数字（0或1）")
        except KeyboardInterrupt:
            print("\n👋 用户取消，程序退出")
            sys.exit(0)
    
    conversation: List[dict] = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 根据模式选择prompt来源
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    try:
        for prompt in prompt_iter:
            # 处理空输入
            if not prompt or not prompt.strip():
                if input_mode == 1:
                    print("⚠️  输入为空，请重新输入")
                    continue
                else:
                    break
            
            setup_seed(2026)  # 固定随机种子以保证可复现性
            
            if input_mode == 0:
                print(f'💬: {prompt}')
            
            # 保留指定数量的历史对话
            conversation = conversation[-args.historys:] if args.historys else []
            conversation.append({"role": "user", "content": prompt})

            # 构建输入模板
            templates = {
                "conversation": conversation, 
                "tokenize": False, 
                "add_generation_prompt": True
            }
            if args.weight == 'reason':
                templates["enable_thinking"] = True  # 仅Reason模型使用
            
            # 准备输入
            if args.weight != 'pretrain':
                inputs = tokenizer.apply_chat_template(**templates)
            else:
                inputs = tokenizer.bos_token + prompt
            
            inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

            # 生成回复
            print('🤖: ', end='', flush=True)
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
            except RuntimeError as e:
                print(f"\n❌ 生成失败: {e}")
                if "out of memory" in str(e).lower():
                    print("💡 建议: 减少max_new_tokens或使用更小的模型")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            
            # 解码回复
            response = tokenizer.decode(
                generated_ids[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            conversation.append({"role": "assistant", "content": response})
            
            # 计算并显示生成速度
            if args.show_speed:
                gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
                elapsed_time = time.time() - st
                if elapsed_time > 0:  # 避免除零错误
                    speed = gen_tokens / elapsed_time
                    print(f'\n[Speed]: {speed:.2f} tokens/s\n')
                else:
                    print('\n')
            else:
                print('\n')
                
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()