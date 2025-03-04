import sys
sys.path.insert(1, "./")
from tokenizer.intuit import load_tokenizer, load_dictionary, find_tokens
import torch

token_mapping = load_dictionary("data/tokenizer_v1.json")["direct"]
print(len(list(token_mapping.keys()))/64**3)

tokenizer, detokenizer = load_tokenizer("data/tokenizer_v1.json")

sample_line = "Hello, please explain to me why earth rotates? okay"
print(find_tokens(sample_line))

tokens = tokenizer(sample_line)
print(tokens)
print(detokenizer(tokens))

with open("data/training_input.txt","r") as f:
    text = f.read()

text_tokens = torch.tensor([int(y) for x in tokenizer(text) for y in x.split(":")],dtype=torch.short)
print(text_tokens[:1000])
