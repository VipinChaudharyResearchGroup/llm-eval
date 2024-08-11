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
    input_masks: torch.Tensor,
    min_prompt_len: int,
    total_len: int,
    eos_token: int,
    method: Literal["greedy", "top_k", "top_p"] = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length=30,
    max_new_tokens=None,
    num_return_sequences=1,
):

    device = input_ids.device
    model.eval()
    model.to(device)

    batch_size = input_ids.size(0)

    eos_reached = torch.tensor([False] * batch_size, device=device)

    prev_pos = 0
    for cur_pos in range(min_prompt_len, total_len):

        logits = model(input_ids[:, prev_pos:cur_pos], prev_pos)

        next_token = select_next_token(logits, method, temperature, top_k, top_p)

        next_token = next_token.reshape(-1)

        next_token = torch.where(
            input_masks[:, cur_pos], input_ids[:, cur_pos], next_token
        )

        input_ids[:, cur_pos] = next_token

        eos_reached |= (~input_masks[:, cur_pos]) & (next_token == eos_token)

        prev_pos = cur_pos
        if all(eos_reached):
            break

    return input_ids
