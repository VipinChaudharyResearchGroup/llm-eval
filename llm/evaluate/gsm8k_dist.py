import json

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import concatenate_datasets, load_dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoTokenizer

distributed_state = PartialState()

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=distributed_state.device, torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    tokenizer.pad_token_id = 0

if model.generation_config.pad_token_id is None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def create_cot_tokens(
    json_file,
    tokenizer,
    n_shots=8,
    reasoning_trigger="",
    final_answer_trigger="",
    idle_trigger="",
):

    assert n_shots <= 8, "n_shots should be less than or equal to 8"

    questions = []
    chains = []
    answers = []

    with open(json_file, "r") as file:
        data = json.load(file)

    for item in data:
        questions.append(item["question"])
        chains.append(" ".join(item["reasoning"]))
        answers.append(item["answer"])

    questions, chains, answers = (
        questions[:n_shots],
        chains[:n_shots],
        answers[:n_shots],
    )

    cot_prompt = ""

    for i in range(n_shots):
        parts = []
        if reasoning_trigger:
            parts.append(reasoning_trigger)
        parts.append(chains[i])
        if final_answer_trigger:
            parts.append(final_answer_trigger)
        parts.append(answers[i])
        if idle_trigger:
            parts.append(idle_trigger)

        response = " ".join(parts)
        cot_prompt += f"Q: {questions[i]}\nA: {response}.\n\n"

    tokenized_cot = tokenizer(
        cot_prompt, padding=False, add_special_tokens=False, return_tensors="pt"
    )

    return cot_prompt, tokenized_cot


def concat_cot_tokens(tokenizer, tokenized_cot, question):
    wrapped_question = f"Q: {question}\nA:"
    # tokenized_question = tokenizer(wrapped_question, return_tensors="pt")
    tokenized_question = tokenizer(
        wrapped_question, padding=True, return_tensors="pt", pad_to_multiple_of=8
    )

    combined_tokens = {
        key: torch.cat([tokenized_cot[key], tokenized_question[key]], dim=1)
        for key in ["input_ids", "attention_mask"]
    }

    return {
        "input_ids": combined_tokens["input_ids"],
        "attention_mask": combined_tokens["attention_mask"],
    }
    # return combined_tokens


def load_gsm8k(subset="test"):
    gsm = load_dataset("openai/gsm8k", "main")

    if subset == "train":
        return gsm["train"]
    elif subset == "test":
        return gsm["test"]
    elif subset == "both":
        gsm_eval = concatenate_datasets([gsm["train"], gsm["test"]])
        gsm_eval = gsm_eval.shuffle(seed=42)
        return gsm_eval


n_shots = 8

# Triggers
reasoning_trigger = "Let's think step by step."
reasoning_trigger = "Let's break it down."

final_answer_trigger = "The answer is"
final_answer_trigger = "Therefore, the final answer is"

idle_trigger = "Let's go to the next question"

cot_prompt, tokenized_cot = create_cot_tokens(
    "./templates/gsm8k_cot.json",
    tokenizer,
    n_shots=n_shots,
    reasoning_trigger=reasoning_trigger,
    final_answer_trigger=final_answer_trigger,
    # idle_trigger=idle_trigger,
)


batch_size = 7

gsm = load_gsm8k(subset="test")

ds_len = len(gsm)

gsm = gsm.batch(batch_size=batch_size)


# Apply padding on the left since we are doing generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"

gsm_test = gsm.map(
    lambda x: tokenizer(x["question"], padding=True, pad_to_multiple_of=8),
)

tokenizer.padding_side = padding_side_default


cot_ids = tokenized_cot["input_ids"].expand(batch_size, -1)
cot_attention = tokenized_cot["attention_mask"].expand(batch_size, -1)


completions_per_process = []

with distributed_state.split_between_processes(
    gsm_test, apply_padding=True
) as batched_prompts:

    for prompt in batched_prompts:

        if len(prompt["input_ids"]) == batch_size:

            batched_input_ids = torch.cat(
                [cot_ids, torch.tensor(prompt["input_ids"])], dim=1
            ).to(distributed_state.device)
            batched_attention_mask = torch.cat(
                [cot_attention, torch.tensor(prompt["attention_mask"])], dim=1
            ).to(distributed_state.device)

        else:
            n = len(prompt["input_ids"])
            batched_input_ids = torch.cat(
                [cot_ids[:n], torch.tensor(prompt["input_ids"])], dim=1
            ).to(distributed_state.device)
            batched_attention_mask = torch.cat(
                [cot_attention[:n], torch.tensor(prompt["attention_mask"])], dim=1
            ).to(distributed_state.device)

        outputs = model.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        completions_per_process.extend(generated_text)


completions_gather = gather_object(completions_per_process)

# Drop duplicates produced by apply_padding in split_between_processes
completions = completions_gather[:ds_len]

distributed_state.print(completions)
