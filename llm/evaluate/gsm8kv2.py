import json
import logging
import re
from datetime import datetime
from pathlib import Path

import csv

import torch
from datasets import concatenate_datasets, load_dataset
from init import init
from transformers import AutoModelForCausalLM, AutoTokenizer


# Triggers
reasoning_trigger = "Let's break it down."
final_answer_trigger = "Therefore, the final answer is"
idle_trigger = "Let's go to the next question."


def generate_template(question, chain, answer):
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    return question, chain, answer


def create_cot_prompt(n_shots=8):

    assert n_shots <= 8, "n_shot should be less than or equal to 8"

    questions, chains, answers = [], [], []

    questions, chains, answers = generate_template(questions, chains, answers)

    questions, chains, answers = (
        questions[:n_shots],
        chains[:n_shots],
        answers[:n_shots],
    )

    cot_prompt = ""
    for i in range(n_shots):
        cot_prompt += f"Q: {questions[i]}\nA: {reasoning_trigger} {chains[i]} {final_answer_trigger} {answers[i]}. {idle_trigger}\n\n"

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


def concat_cot_prompt(question, n_shots=8):
    cot_prompt = create_cot_prompt(n_shots)
    prompt = cot_prompt + f"Q: {question}\nA:"
    return prompt


def load_model(checkpoint):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # for key in tokenizer.special_tokens_map:
    #     print(f"{key}: {tokenizer.special_tokens_map[key]}")
    #     print(f"ID: {tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map[key])}")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

    # gen_conf = model.generation_config
    # gen_conf = gen_conf.to_dict()
    # for key in gen_conf:
    #     print(f"{key}: {gen_conf[key]}")

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

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


# def clean_response(response):

#     response = response.replace("\n", " ").lower().strip()

#     first_response = response.split(final_answer_trigger.lower())

#     if len(first_response) > 1:  # final answer trigger found

#         first_response = first_response[1].split("\n")[0].strip()[:-1]

#         reg_rum_pattern = r"(\d+(\.\d+)?)"

#         match = re.search(reg_rum_pattern, first_response)

#         if match:
#             num_str = match.group(1)
#             num = float(num_str) if "." in num_str else int(num_str)
#             return num
#         else:
#             return None

#     else:
#         expected = response.split("therefore")[-1]
#         numbers = re.findall(r"\d+\.?\d*", expected)
#         parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]
#         num_parsed = len(parsed_numbers)

#         if num_parsed == 1:
#             return parsed_numbers[0]
#         elif num_parsed > 1:
#             return parsed_numbers[-1]
#         else:
#             return None


# def clean_response(response, final_answer_trigger="final answer"):

#     response = response.replace("\n", " ").lower().strip()

#     first_response = response.split(final_answer_trigger)
#     reg_num_pattern = r"(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"

#     if len(first_response) > 1:
#         # print("one")
#         first_response = first_response[1].split("\n")[0].strip()[:-1]

#         # reg_num_pattern = r"(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
#         # match = re.search(reg_num_pattern, first_response)

#         parsed_numbers = re.findall(reg_num_pattern, first_response)

#         num_parsed = len(parsed_numbers)

#         if num_parsed:
#             return float(parsed_numbers[-1].replace("$", "").replace(",", ""))

#             # currency_str = match.group(1)
#             # cleaned_currency_str = currency_str.replace('$', '').replace(',', '')
#             # num = float(cleaned_currency_str)
#             # return num
#         else:
#             return None

#     else:
#         # print("two")

#         expected = response.split("therefore")[-1].strip()

#         # Updated regular expression to capture numbers with currency symbols and commas
#         # reg_num_pattern = r"(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
#         parsed_numbers = re.findall(reg_num_pattern, expected)

#         num_parsed = len(parsed_numbers)

#         if num_parsed:

#             return float(parsed_numbers[-1].replace("$", "").replace(",", ""))
#         else:
#             return None


# def clean_response(response):

#     response = response.lower().strip().split("\n")[0]

#     first_response = response.split(final_answer_trigger.lower())

#     if len(first_response) > 1:  # final answer trigger found

#         first_response = first_response[1].split("\n")[0].strip()[:-1]
#         # first_response = first_response[1]

#         # print(first_response)

#         expected = first_response.replace("$", "").replace(",", "")

#         numbers = re.findall(r"-?\d+\.?\d*", expected)

#         parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

#         num_parsed = len(parsed_numbers)

#         if num_parsed > 0:
#             return parsed_numbers[-1]
#         else:
#             return None

#     else:
#         # print("two")
#         expected = response.split("therefore")[-1]

#         expected = expected.replace("$", "").replace(",", "")

#         numbers = re.findall(r"-?\d+\.?\d*", expected)

#         parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

#         num_parsed = len(parsed_numbers)

#         if num_parsed > 0:
#             return parsed_numbers[-1]
#         else:
#             return None


def clean_response(response):

    response_ = response.lower().strip().split("\n")[0]

    first_response = response_.split(final_answer_trigger.lower())

    if len(first_response) > 1:  # final answer trigger found
        expected = first_response[1].split("\n")[0].strip()[:-1]
    else:
        expected = response_.split("therefore")[-1]

    expected = expected.replace("$", "").replace(",", "")

    numbers = re.findall(r"-?\d+\.?\d*", expected)

    parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

    num_parsed = len(parsed_numbers)

    if num_parsed > 0:
        return parsed_numbers[-1]
    else:
        exception_candidates = [
            "there is no",
            "there are no",
            "there will be no",
            "there will not be",
            "there will not be",
        ]

        for candidate in exception_candidates:
            if candidate in expected:
                return 0

        cle = response.split("Q:")[0].split("\n")
        l = len(cle)

        for i in range(l):
            if len(cle[l - i - 1]) > 0:
                x = cle[l - i - 1].lower().split(final_answer_trigger.lower())
                expected = x[-1].replace("$", "").replace(",", "")

                numbers = re.findall(r"-?\d+\.?\d*", expected)
                parsed_numbers = [
                    float(num) if "." in num else int(num) for num in numbers
                ]
                num_parsed = len(parsed_numbers)
                if num_parsed > 0:
                    return parsed_numbers[-1]

        return None


def evaluate(
    model,
    tokenizer,
    generate_kwargs,
    n_shots=8,
    subset="test",
    iterations=None,
    save_response=False,
    debug=False,
):

    readable_responses, corrects, input_length_total, input_length_avg = 0, 0, 0, 0

    gsm_eval = load_gsm8k(subset=subset)
    num_questions = len(gsm_eval)

    csv_file_path_successful = f"{save_response}/successful_responses.csv"
    csv_file_path_unsuccessful = f"{save_response}/unsuccessful_responses.csv"

    columns = ["index", "question", "resp", "resp_parsed", "ans", "ans_parsed"]

    with open(csv_file_path_successful, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["correct"])

    with open(csv_file_path_unsuccessful, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["error"])

    for i in range(num_questions) if iterations is None else range(iterations):

        question, answer_truth_ = gsm_eval[i]["question"], gsm_eval[i]["answer"]
        answer_truth = answer_truth_.split(" ")[-1].replace(",", "")

        prompt = concat_cot_prompt(question, n_shots)

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
            if debug:
                print(f"Inference prompt: {question} \n")
                print(f"Response: {response} \n")
                print(f"Predicted answer: {final_answer_prediction} \n")
                print(f"True answer: {final_answer_truth} \n")

    if readable_responses == 0:
        accuracy, input_length_avg = 0, 0

    else:
        accuracy = corrects / readable_responses
        input_length_avg = input_length_total / readable_responses

    print(f"Accuracy: {accuracy}")

    if save_response:

        cot_prompt = concat_cot_prompt("the next question comes here", n_shots)

        with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

        with open(csv_file_path_successful, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    return accuracy, readable_responses, input_length_avg


def evaluate_init(checkpoint):

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = checkpoint.replace("/", "_")
    output_dir = Path(rf"output/{checkpoint_path}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 256,
    }

    model, tokenizer = load_model(checkpoint)

    accuracy, readable_responses, input_length_avg = evaluate(
        model,
        tokenizer,
        generate_kwargs,
        n_shots=8,
        # subset="test",
        iterations=None,
        # iterations=1,
        save_response=output_dir,
    )

    config = model.config
    config = config.to_dict()

    results = {
        "model": checkpoint,
        "accuracy": accuracy,
        "num_responses": readable_responses,
        "generate_kwargs": generate_kwargs,
        "config": config,
        "input_tokens_avg": input_length_avg,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    config = init()
    print(config.device)
    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoints = [
        # "allenai/OLMo-1.7-7B-hf",
        # "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Meta-Llama-3-8B",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "mistralai/Mistral-7B-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.2",
        # "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    for checkpoint in checkpoints:
        try:
            print(f"Running {checkpoint} \n")
            evaluate_init(checkpoint)
        except Exception as e:
            logging.error(f"Error: in {checkpoint} \n {e}")
        finally:
            torch.cuda.empty_cache()
