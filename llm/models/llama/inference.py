import json
from pathlib import Path

import torch
from Config import ModelArgs
from init import init
from model_single_node import Llama2
from text_generation import generate
from tokenizer import Tokenizer

config = init()

tokenizer = Tokenizer(f"{config.base_dir}/.cache/meta_llama2/tokenizer.model")


def load_config(
    vocab_size,
    llama_path=Path(f"{config.base_dir}/.cache/meta_llama2/llama-2-7b/"),
):
    with open(llama_path / "params.json", "r") as f:  # Load the params
        params = json.loads(f.read())

    model_args = ModelArgs(
        device="cuda",
        **params,
    )
    model_args.vocab_size = vocab_size

    return model_args


model_args = load_config(tokenizer.vocab_size)

model = Llama2.from_pretrained(
    weight_path=Path(f"{config.base_dir}/.cache/meta_llama2/llama-2-7b/"),
    params=model_args,
)

torch.cuda.manual_seed(42)
torch.manual_seed(42)

prompt = ["Hello, I'm a language model", "Simply put, I'm a text generation model"]

input_ids, input_masks, total_len, min_prompt_len = tokenizer.encode_prompts(
    prompts=prompt, max_gen_len=128, params=model.params, device=config.device
)

output_ids = generate(
    model=model,
    input_ids=input_ids,
    input_masks=input_masks,
    min_prompt_len=min_prompt_len,
    total_len=total_len,
    eos_token=tokenizer.eos_token,
    # method="greedy",
    method="top_p",
    # method="top_k",
    temperature=0.6,
    top_k=50,
    top_p=0.8,
    max_length=60,
    max_new_tokens=None,
    num_return_sequences=1,
)

decoded_output = tokenizer.decode_tokens(output_ids)

for text in decoded_output:
    print("-" * 50, "\n")
    print(text)
