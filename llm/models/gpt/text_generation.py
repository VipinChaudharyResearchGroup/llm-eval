from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    input_len = input_ids.shape[1]
    num_new_tokens = max(max_new_tokens, max_length - input_len)

    model.eval()
    model.to(input_ids.device)

    for _ in range(num_new_tokens):

        logits = model(input_ids)  # (batch_size, sequence_length, vocab_size)
        # next_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        next_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        if temperature != 1.0:
            next_logits /= temperature

        next_probs = F.softmax(next_logits, dim=-1)  # (batch_size, vocab_size)
        # print(next_probs.sum(dim=-1))   # sum of probabilities should be 1

        if method == "greedy":
            # torch.max returns (values, indices)
            # using keepdim=True, no need to unsqueeze the tensor
            # _, next_token = torch.max(next_probs, dim=-1, keepdim=True) # the same functionality
            _, next_token = next_probs.max(dim=-1, keepdim=True)

        else:
            if method == "top_k":
                # torch.topk returns (values, indices)
                # probs, probs_indices = torch.topk(input=next_probs, k=top_k, dim=-1)
                probs, probs_indices = next_probs.topk(k=top_k, dim=-1)

            elif method == "top_p":

                probs, probs_indices = next_probs.sort(
                    descending=True, dim=-1
                )  # (batch_size, vocab_size)
                cumulative_probs = probs.cumsum(dim=-1)  # (batch_size, vocab_size)
                mask = cumulative_probs - probs > top_p
                probs[mask] = 0.0
                # should be normalized since torch.multinomial expects normalized probabilities
                # probs.div_(probs.sum(dim = -1, keepdim = True) + 1e-6)
                probs.div_(probs.sum(dim=-1, keepdim=True))

            else:
                raise ValueError(
                    "Invalid method or missing required argument (top_p or top_k)."
                )

            idx_sample = torch.multinomial(input=probs, num_samples=1)

            next_token = torch.gather(input=probs_indices, dim=-1, index=idx_sample)

        input_ids = torch.cat(
            [input_ids, next_token], dim=-1
        )  # (batch_size, sequence_length + 1)

    return input_ids
