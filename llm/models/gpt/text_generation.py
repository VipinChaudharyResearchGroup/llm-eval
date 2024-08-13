from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_next_token(
    logits: torch.Tensor,
    method: Literal["greedy", "top_k", "top_p"] = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    next_logits = logits[:, -1, :]

    if method == "greedy":
        _, next_token = next_logits.max(dim=-1, keepdim=True)

    else:
        next_logits /= temperature  # default temperature = 1.0
        next_probs = F.softmax(next_logits, dim=-1)

        if method == "top_k":
            probs, probs_indices = next_probs.topk(k=top_k, dim=-1)

        elif method == "top_p":

            probs, probs_indices = next_probs.sort(descending=True, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)
            mask = cumulative_probs - probs > top_p
            probs[mask] = 0.0
            probs /= probs.sum(dim=-1, keepdim=True)

        else:
            raise ValueError(
                "Invalid method or missing required argument (top_p or top_k)."
            )

        idx_sample = torch.multinomial(input=probs, num_samples=1)

        next_token = torch.gather(input=probs_indices, dim=-1, index=idx_sample)

    return next_token

@torch.inference_mode()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    method: Literal["greedy", "top_k", "top_p"] = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length=30,
    max_new_tokens=None,
    num_return_sequences=1,
):
    """
        Generate a sequence of tokens using the model.
        1. Initial Input: The process begins with an initial sequence of tokens represented by input_ids, which typically has a shape (batch_size, sequence_length).
        2. Token-by-Token Generation: The model generates new tokens one at a time. After generating each token, it appends the token to the input sequence and uses the updated sequence to generate the next token.
        3. Sequence Continuation: This process continues until the sequence reaches a specified maximum length, a stop token is generated, or another stopping criterion is met.

    Args:
        input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) and dtype torch.int64 (LongTensor).
        max_length (int): The maximum length of the sequence to be generated.
        num_return_sequences (int): The number of independently computed returned sequences for each element in the batch.
        do_sample (bool): If set to False greedy decoding is used. Otherwise, sampling is used.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filter

    Returns:
        torch.Tensor: A tensor of shape (batch_size, max_length) and dtype torch.int64 (LongTensor).

    """
    # max_new_token = max_new_token or max_length # refactor this later
    # s.t.
    # max_new_tokens + input_ids.shape[1] = max_length

    model.eval()
    model.to(input_ids.device)

    input_len = input_ids.shape[1]
    num_new_tokens = max(max_new_tokens, max_length - input_len)

    for _ in range(num_new_tokens):

        logits = model(input_ids)  # (batch_size, sequence_length, vocab_size)
        # next_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        next_token = select_next_token(logits, method, temperature, top_k, top_p)

        input_ids = torch.cat(
            [input_ids, next_token], dim=-1
        )  # (batch_size, sequence_length + 1)

    return input_ids
