import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])


print(len(preprocessed))


# [10]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)


# [11]

vocab = {token:integer for integer,token in enumerate(all_words)}

# [12]

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

    
