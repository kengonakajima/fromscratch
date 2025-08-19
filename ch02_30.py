import importlib.metadata
import tiktoken


import torch
print("PyTorch version:", torch.__version__)

print("tiktoken version:", importlib.metadata.version("tiktoken"))

# [27], [28]
tokenizer = tiktoken.get_encoding("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# [31]
enc_sample = enc_text[50:]

# [32]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")


# [33]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)
    
# [34]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    
