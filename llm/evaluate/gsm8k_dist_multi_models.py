import json

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import concatenate_datasets, load_dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoTokenizer
from init import init
import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset
from evaluate.gsm8k_parse_ans import clean_response
from helpers.profiling import experiment
from init import init
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoints to evaluate")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["meta-llama/Llama-2-7b-hf"],
        help="List of checkpoints to use",
    )
    return parser.parse_args()


batch_size = 7


def input(tokenizer, batch_size=8):
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

    return gsm_test, cot_ids, cot_attention, ds_len


def run(model_name, batch_size=8):
    distributed_state = PartialState()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=distributed_state.device, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    gsm_test, cot_ids, cot_attention, ds_len = input(tokenizer, batch_size=batch_size)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

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

    completions = completions_gather[:ds_len]

    distributed_state.print(completions)


def main(seed=None):

    config = init() if seed is None else init(seed=seed)
    print(config.device)

    args = parse_args()
    checkpoints = args.checkpoints
    # checkpoints = [
    #     "meta-llama/Llama-2-7b-hf",
    #     "meta-llama/Llama-2-7b-chat-hf",
    #     "meta-llama/Meta-Llama-3-8B",
    #     "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "mistralai/Mistral-7B-v0.1",
    #     "mistralai/Mistral-7B-Instruct-v0.1",
    #     "mistralai/Mistral-7B-Instruct-v0.2",
    #     "mistralai/Mistral-7B-Instruct-v0.3",
    #     "allenai/OLMo-1.7-7B-hf",
    # ]

    # logging.basicConfig(
    #     filename="./logs/running.log",
    #     filemode="a",
    #     level=logging.ERROR,
    #     format="%(asctime)s %(levelname)s %(message)s",
    # )

    for checkpoint in checkpoints:
        try:
            print(f"Running {checkpoint} \n")
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            checkpoint_path = checkpoint.replace("/", "_")
            output_dir = Path(rf"output/GSM/multi_{checkpoint_path}_{now}")
            output_dir.mkdir(parents=True, exist_ok=True)

            @experiment(
                experiment_name=f"gsm8k_dist_{checkpoint_path}",
                num_experiments=1,
                save_profile=False,
            )
            def experiment_main():
                return run(checkpoint, output_dir)

            profile = experiment_main()

            profile["seed"] = seed

            with open(output_dir / "results.json", "w") as f:
                json.dump(profile, f)

        except Exception as e:
            logging.error(f"Error: in {checkpoint} \n {e}")
        finally:
            torch.cuda.empty_cache()
