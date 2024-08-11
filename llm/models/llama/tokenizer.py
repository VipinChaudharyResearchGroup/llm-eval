import torch
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size()
        self.eos_token = self.tokenizer.eos_id()

    def encode_prompts(self, prompts, max_gen_len, params, device):
        batch_size = len(prompts)
        assert (
            batch_size <= params.max_batch_size
        ), f"Input batch size, {batch_size}, exceeds the maximum batch size, {params.max_batch_size}"

        tokens_batch = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        max_prompt_len = max(map(len, tokens_batch))
        min_prompt_len = min(map(len, tokens_batch))
        assert (
            max_prompt_len <= params.max_seq_len
        ), f"Max prompt length, {max_prompt_len}, exceeds the maximum sequence length, {params.max_seq_len}"

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id()

        input_ids = torch.stack(
            [
                torch.tensor(
                    tokens + [pad_id] * (total_len - len(tokens)), dtype=torch.long
                )
                for tokens in tokens_batch
            ]
        ).to(device)

        input_masks = input_ids != pad_id
        return input_ids, input_masks, total_len, min_prompt_len

    def decode_tokens(self, tokens: torch.Tensor):
        out_text = []
        for token_id in tokens.tolist():
            if self.tokenizer.eos_id in token_id:
                eos_idx = token_id.index(self.tokenizer.eos_id)
                token_id = token_id[:eos_idx]
            out_text.append(self.tokenizer.decode(token_id))
        return out_text
