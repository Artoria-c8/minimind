import time
import os
import argparse
import random
import warnings

warnings.filterwarnings("ignore")


def _ensure_pad_token(tokenizer) -> None:
    # 一些 tokenizer 没有 pad_token，会导致 generate/attention_mask 行为不稳定
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0


def _resolve_ckpt_path(save_dir: str, weight: str, hidden_size: int, use_moe: int) -> str:
    moe_suffix = "_moe" if use_moe else ""
    filename = f"{weight}_{hidden_size}{moe_suffix}.pth"
    return os.path.join(save_dir, filename)

def init_model(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from model.model_lora import apply_lora, load_lora
    from trainer.trainer_utils import get_model_params

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    _ensure_pad_token(tokenizer)
    if args.load_from == 'model':
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        ckp = _resolve_ckpt_path(args.save_dir, args.weight, args.hidden_size, args.use_moe)
        if not os.path.exists(ckp):
            raise FileNotFoundError(f'未找到模型权重文件：{ckp}')
        state = torch.load(ckp, map_location='cpu')
        model.load_state_dict(state, strict=True)
        if args.lora_weight and str(args.lora_weight).lower() not in {'none', 'null'}:
            apply_lora(model)
            lora_path = os.path.join(args.save_dir, 'lora', f'{args.lora_weight}_{args.hidden_size}.pth')
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f'未找到LoRA权重文件：{lora_path}')
            load_lora(model, lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            trust_remote_code=True,
            torch_dtype='auto',
        )
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录（原生torch权重模式下使用）")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default=None, type=str, help="LoRA权重名称（不传表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='auto', type=str, help="运行设备（auto/cpu/cuda）")
    parser.add_argument('--seed', default=2026, type=int, help="随机种子（-1表示每轮随机）")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit("缺少依赖：torch。请先安装 requirements.txt 中的依赖后再运行。") from e

    from transformers import TextStreamer
    from trainer.trainer_utils import setup_seed

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    conversation = []
    model, tokenizer = init_model(args)
    try:
        input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    except (ValueError, EOFError):
        input_mode = 0
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        if args.seed == -1:
            setup_seed(random.randint(0, 2**31 - 1))
        else:
            setup_seed(args.seed)
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
            )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()