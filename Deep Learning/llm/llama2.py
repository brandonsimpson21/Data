from transformers import AutoTokenizer
import transformers
import torch


n_params = 7
# n_params = 13

model = f"meta-llama/Llama-2-{n_params}b-chat-hf"
prompt = "my favorite food is cheesecake, what other kinds of dessert may i like?\n"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=300,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
