from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1.7-7B-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")

message = ["My name is "]
inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False)
response = olmo.generate(
    **inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
