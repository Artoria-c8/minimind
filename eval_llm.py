import time
import argparse
import random
import warnings
import os
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    初始化模型和分词器
    
    Args:
        args: 命令行参数对象，包含模型加载相关配置
        
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: 返回模型和分词器的元组
        
    Raises:
        FileNotFoundError: 当模型权重文件不存在时
        RuntimeError: 当模型加载失败时
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    except Exception as e:
        raise RuntimeError(f"加载分词器失败: {e}")
    
    if 'model' in args.load_from:
        try:
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling
            ))
            moe_suffix = '_moe' if args.use_moe else ''
            ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
            
            if not os.path.exists(ckp):
                raise FileNotFoundError(f"模型权重文件不存在: {ckp}")
            
            model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
            
            if args.lora_weight != 'None':
                lora_path = f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'
                if not os.path.exists(lora_path):
                    raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
                apply_lora(model)
                load_lora(model, lora_path)
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"从transformers格式加载模型失败: {e}")
    
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def get_user_input() -> Optional[str]:
    """
    获取用户输入，用于手动输入模式
    
    Returns:
        Optional[str]: 用户输入的文本，如果输入为空则返回None
    """
    try:
        user_input = input('💬: ')
        return user_input if user_input.strip() else None
    except (EOFError, KeyboardInterrupt):
        return None


def main() -> None:
    """
    主函数：MiniMind模型推理与对话的主入口
    """
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
    
    # 默认测试提示词
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
    
    try:
        conversation = []
        model, tokenizer = init_model(args)
        
        try:
            input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
        except (ValueError, EOFError, KeyboardInterrupt):
            print("输入无效，使用默认模式（自动测试）")
            input_mode = 0
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 根据输入模式选择提示词来源
        if input_mode == 0:
            prompt_iter = iter(DEFAULT_PROMPTS)
        else:
            # 手动输入模式：使用生成器函数
            def prompt_generator():
                while True:
                    prompt = get_user_input()
                    if prompt is None:
                        break
                    yield prompt
            prompt_iter = prompt_generator()
        
        for prompt in prompt_iter:
            setup_seed(2026)  # 或 setup_seed(random.randint(0, 2048))
            if input_mode == 0:
                print(f'💬: {prompt}')
            
            # 管理对话历史
            conversation = conversation[-args.historys:] if args.historys else []
            conversation.append({"role": "user", "content": prompt})

            # 准备模板和输入
            templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
            if args.weight == 'reason':
                templates["enable_thinking"] = True  # 仅Reason模型使用
            
            if args.weight != 'pretrain':
                prompt_text = tokenizer.apply_chat_template(**templates)
            else:
                prompt_text = tokenizer.bos_token + prompt if tokenizer.bos_token else prompt
            
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(args.device)

            print('🤖: ', end='', flush=True)
            start_time = time.time()
            
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
                response = tokenizer.decode(
                    generated_ids[0][len(inputs["input_ids"][0]):], 
                    skip_special_tokens=True
                )
                conversation.append({"role": "assistant", "content": response})
                gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
                
                if args.show_speed:
                    elapsed_time = time.time() - start_time
                    speed = gen_tokens / elapsed_time if elapsed_time > 0 else 0
                    print(f'\n[Speed]: {speed:.2f} tokens/s\n\n')
                else:
                    print('\n\n')
            except Exception as e:
                print(f'\n生成失败: {e}\n\n')
                continue
                
    except Exception as e:
        print(f"程序运行出错: {e}")
        raise

if __name__ == "__main__":
    main()