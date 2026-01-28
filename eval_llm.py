import time
import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings("ignore")

# 常量定义
DEFAULT_SEED = 2026
DEFAULT_PROMPTS = [
    '你有什么特长？',
    '为什么天空是蓝色的',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '如果明天下雨，我应该如何出门',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '推荐一些中国的美食'
]


def init_model(args) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    初始化模型和分词器
    
    Args:
        args: 命令行参数对象
        
    Returns:
        Tuple[model, tokenizer]: 模型和分词器
        
    Raises:
        FileNotFoundError: 当模型文件不存在时
        RuntimeError: 当模型加载失败时
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    except Exception as e:
        raise RuntimeError(f"无法加载分词器从路径: {args.load_from}, 错误: {e}")
    
    if 'model' in args.load_from:
        try:
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling
            ))
            moe_suffix = '_moe' if args.use_moe else ''
            ckp_path = Path(args.save_dir) / f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'
            
            if not ckp_path.exists():
                raise FileNotFoundError(f"模型权重文件不存在: {ckp_path}")
            
            model.load_state_dict(torch.load(str(ckp_path), map_location=args.device), strict=True)
            
            if args.lora_weight != 'None':
                apply_lora(model)
                lora_path = Path(args.save_dir) / 'lora' / f'{args.lora_weight}_{args.hidden_size}.pth'
                if not lora_path.exists():
                    raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
                load_lora(model, str(lora_path))
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"无法从transformers加载模型: {args.load_from}, 错误: {e}")
    
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def validate_args(args) -> None:
    """
    验证命令行参数
    
    Args:
        args: 命令行参数对象
        
    Raises:
        ValueError: 当参数无效时
    """
    if args.historys < 0:
        raise ValueError("historys 参数必须 >= 0")
    if args.historys > 0 and args.historys % 2 != 0:
        raise ValueError("historys 参数必须为偶数（0表示不携带历史）")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature 参数必须在 0-1 之间")
    if args.top_p < 0 or args.top_p > 1:
        raise ValueError("top_p 参数必须在 0-1 之间")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens 参数必须 > 0")
    if args.hidden_size <= 0:
        raise ValueError("hidden_size 参数必须 > 0")
    if args.num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers 参数必须 > 0")


def get_input_mode() -> int:
    """
    获取用户输入模式
    
    Returns:
        int: 输入模式（0=自动测试，1=手动输入）
    """
    while True:
        try:
            mode = input('[0] 自动测试\n[1] 手动输入\n请选择: ').strip()
            if mode in ['0', '1']:
                return int(mode)
            print("❌ 无效输入，请输入 0 或 1")
        except (EOFError, KeyboardInterrupt):
            print("\n\n程序已取消")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 输入错误: {e}")


def main():
    """主函数：MiniMind模型推理与对话"""
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, 
                       help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, 
                       help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, 
                       help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, 
                       help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                       help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                       help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', 
                       help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, 
                       help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, 
                       help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, 
                       help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, choices=[0, 1], 
                       help="显示decode速度（0=否，1=是）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                       type=str, help="运行设备")
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int, 
                       help="随机种子（用于可复现性）")
    
    args = parser.parse_args()
    
    # 验证参数
    try:
        validate_args(args)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        sys.exit(1)
    
    # 初始化模型
    try:
        print("🔄 正在加载模型...")
        model, tokenizer = init_model(args)
        print("✅ 模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 获取输入模式
    input_mode = get_input_mode()
    print()
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []
    
    # 创建提示迭代器
    if input_mode == 0:
        prompt_iter = DEFAULT_PROMPTS
    else:
        def manual_input():
            while True:
                try:
                    prompt = input('💬: ').strip()
                    if not prompt:
                        break
                    yield prompt
                except (EOFError, KeyboardInterrupt):
                    print("\n\n程序已退出")
                    break
        prompt_iter = manual_input()
    
    # 主循环
    for prompt in prompt_iter:
        try:
            setup_seed(args.seed)
            if input_mode == 0:
                print(f'💬: {prompt}')
            
            # 管理对话历史
            if args.historys > 0:
                conversation = conversation[-args.historys:]
            else:
                conversation = []
            
            conversation.append({"role": "user", "content": prompt})

            # 准备输入
            templates = {
                "conversation": conversation, 
                "tokenize": False, 
                "add_generation_prompt": True
            }
            if args.weight == 'reason':
                templates["enable_thinking"] = True  # 仅Reason模型使用
            
            if args.weight != 'pretrain':
                input_text = tokenizer.apply_chat_template(**templates)
            else:
                input_text = (tokenizer.bos_token or '') + prompt
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)

            # 生成回复
            print('🤖: ', end='', flush=True)
            st = time.time()
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
            
            response = tokenizer.decode(
                generated_ids[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            conversation.append({"role": "assistant", "content": response})
            
            # 显示速度信息
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            elapsed_time = time.time() - st
            if args.show_speed:
                print(f'\n[Speed]: {gen_tokens / elapsed_time:.2f} tokens/s\n')
            else:
                print('\n')
                
        except KeyboardInterrupt:
            print("\n\n⚠️  生成已中断")
            continue
        except Exception as e:
            print(f"\n❌ 生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()