import tiktoken
import torch
from GPT2 import GPT2
from helpers.profiling import experiment
from init import init
from text_generation import generate

conf = init(seed=42)

enc = tiktoken.get_encoding("gpt2")


def generate_input_ids(prompt: str, batch_size: int):
    tokens = enc.encode(prompt)  # (sequence_length,)
    tokens = torch.tensor(tokens, dtype=torch.long, device=conf.device)
    input_ids = tokens.unsqueeze(0).repeat(
        batch_size, 1
    )  # (batch_size, sequence_length)
    return input_ids


@experiment(experiment_name="contiguous", num_experiments=10, save_profile=True)
def main():

    input_ids = generate_input_ids(
        "Simply put, the theory of relativity states that", batch_size=2
    )

    model = GPT2("gpt2").from_pretrained()

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        method="top_k",
        # method="top_p",
        # method="greedy",
        top_k=50,
        # top_p=0.9,
        # max_length=30,
        max_new_tokens=128,
        num_return_sequences=2,
        # temperature=0.6,
    )

    for decoded_output in enc.decode_batch(output_ids.tolist()):
        print("-" * 50, "\n")
        print(decoded_output)


if __name__ == "__main__":
    main()
