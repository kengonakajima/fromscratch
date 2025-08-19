import importlib.metadata
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

# [27], [28]
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)




# [29]

strings = tokenizer.decode(integers)

print(strings)

