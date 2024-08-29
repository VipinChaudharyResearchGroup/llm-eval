import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from evaluate.gsm8k_parse_ans import clean_response
from init import init
from helpers.profiling import profiler
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


def create_cot_prompt(
    json_file, n_shots=8, reasoning_trigger="", final_answer_trigger="", idle_trigger=""
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

    return cot_prompt


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


def concat_cot_prompt(cot_prompt, question):
    prompt = cot_prompt + f"Q: {question}\nA:"
    return prompt


def load_model(checkpoint):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.penalty_alpha = None
    model.generation_config.do_sample = False

    model.eval()

    return model, tokenizer


def generate_response(
    prompt, model, tokenizer, generate_kwargs, remove_input_prompt=True
):
    input = tokenizer(
        prompt, padding=False, add_special_tokens=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**input, **generate_kwargs)
    output = output[0]

    input_prompt_len = input["input_ids"].shape[1]

    if remove_input_prompt:
        output = output[input_prompt_len:]

    response = tokenizer.decode(output, skip_special_tokens=True)

    return response, input_prompt_len


def evaluate(
    model,
    tokenizer,
    generate_kwargs,
    n_shots=8,
    subset="test",
    iterations=None,
    save_response=False,
):

    # Triggers
    # reasoning_trigger = "Let's think step by step."
    reasoning_trigger = "Let's break it down."

    final_answer_trigger = "The answer is"
    # final_answer_trigger = "Therefore, the final answer is"

    # idle_trigger = "Let's go to the next question"

    cot_prompt = create_cot_prompt(
        "templates/gsm8k_cot.json",
        n_shots=n_shots,
        reasoning_trigger=reasoning_trigger,
        final_answer_trigger=final_answer_trigger,
        # idle_trigger=idle_trigger,
    )

    readable_responses, corrects, input_length_total, input_length_avg = 0, 0, 0, 0.0

    gsm_eval = load_gsm8k(subset=subset)
    num_questions = len(gsm_eval)

    csv_file_path_successful = f"{save_response}/successful_responses.csv"
    csv_file_path_unsuccessful = f"{save_response}/unsuccessful_responses.csv"

    columns = ["index", "question", "resp", "resp_parsed", "ans", "ans_parsed"]

    with open(csv_file_path_successful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["correct"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["error"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    for i in range(num_questions) if iterations is None else range(iterations):

        question, answer_truth_ = gsm_eval[i]["question"], gsm_eval[i]["answer"]
        answer_truth = answer_truth_.split(" ")[-1].replace(",", "")

        prompt = concat_cot_prompt(cot_prompt, question)

        print(f"Iteration, {i} \n")

        response, input_prompt_len = generate_response(
            prompt, model, tokenizer, generate_kwargs
        )

        final_answer_prediction = clean_response(response)

        if isinstance(final_answer_prediction, (int, float)):

            try:

                final_answer_truth = (
                    float(answer_truth) if "." in answer_truth else int(answer_truth)
                )

                if final_answer_prediction == final_answer_truth:
                    corrects += 1

                if save_response:
                    with open(csv_file_path_successful, mode="a", newline="") as file:
                        writer = csv.writer(file)

                        writer.writerow(
                            [
                                i,
                                question,
                                response,
                                final_answer_prediction,
                                answer_truth_,
                                final_answer_truth,
                                final_answer_prediction == final_answer_truth,
                            ]
                        )

                readable_responses += 1
                input_length_total += input_prompt_len

            except ValueError as e:
                print(f"Error: Value Error in iteration {i} \n")
                with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            i,
                            question,
                            response,
                            final_answer_prediction,
                            answer_truth_,
                            answer_truth,
                            f"Cannot convert true answer, {e}",
                        ]
                    )

        else:
            print(f"Error: Value Error in iteration {i} \n")
            with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        i,
                        question,
                        response,
                        final_answer_prediction,
                        answer_truth_,
                        answer_truth,
                        f"Error in parsing response",
                    ]
                )

    if readable_responses == 0:
        accuracy, input_length_avg = 0, 0

    else:
        accuracy = corrects / readable_responses
        input_length_avg = input_length_total / readable_responses

    print(f"Accuracy: {accuracy}")

    return accuracy, readable_responses, input_length_avg, num_questions


def evaluate_init(checkpoint, seed=None):

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = checkpoint.replace("/", "_")
    output_dir = Path(rf"output/GSM/single_{checkpoint_path}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 256,
        "do_sample": False,
        # "temperature": 0.6,
        # "top_k": 50,
    }
    n_shots = 8

    model, tokenizer = load_model(checkpoint)

    accuracy, readable_responses, input_length_avg, num_questions = evaluate(
        model,
        tokenizer,
        generate_kwargs,
        n_shots=n_shots,
        # subset="test",
        # iterations=None,
        # iterations=1,
        save_response=output_dir,
    )

    config = model.config
    config = config.to_dict()

    results = {
        "model": checkpoint,
        "accuracy": accuracy,
        "num_responses": readable_responses,
        "num_questions": num_questions,
        "num_shots": n_shots,
        "generate_kwargs": generate_kwargs,
        "config": config,
        "input_tokens_avg": input_length_avg,
        "seed": seed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)


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

    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for checkpoint in checkpoints:
        try:
            print(f"Running {checkpoint} \n")
            evaluate_init(checkpoint, seed=seed)
        except Exception as e:
            logging.error(f"Error: in {checkpoint} \n {e}")
        finally:
            torch.cuda.empty_cache()


# Single run, profiling
if __name__ == "__main__":

    @profiler
    def profiling_main():
        return main()

    profile = profiling_main()

    print(profile)
