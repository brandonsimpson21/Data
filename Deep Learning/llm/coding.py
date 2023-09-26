from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch


def codellama(
    model="codellama/CodeLlama-7b-hf",
    prompt="import socket\n\ndef ping_exponential_backoff(host: str):",
    max_length=200,
    top_k=10,
    temp=0.1,
    top_p=0.95,
    num_rtn_seq=1,
):
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
        top_k=top_k,
        temperature=temp,
        top_p=top_p,
        num_return_sequences=num_rtn_seq,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )

    return sequences


def stable_code(
    text="import torch\nimport torch.nn as nn",
    model="stabilityai/stablecode-completion-alpha-3b",
    max_new_tokens=48,
    temp=0.2,
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    inputs = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        model.cuda()
        inputs.to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


if __name__ == "__main__":
    sequences = codellama()
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    sequences = stable_code()
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
