import sys
sys.path.insert(1, "./")
import ilm

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1234)

# Tokenizer
token_mapping = ilm.load_dictionary("data/tokenizer_v1.json")["direct"]
print("Percent of code used =", round(len(list(token_mapping.keys()))/64**3,2))

tokenizer, detokenizer = ilm.load_tokenizer("data/tokenizer_v1.json")

sample_line = "Hello, please explain to me why earth rotates? okay"
print("find_tokens =", ilm.find_tokens(sample_line))
tokens = tokenizer(sample_line)
print("token codes =", tokens)
print("token words =", detokenizer(tokens))


# Tokenize dataset
with open("data/training_input.txt","r") as f:
    raw_text = f.read()

manager = ilm.TrainingManager(raw_text, tokenizer)

x, y = manager.get_batch("train")
ilmodel = ilm.IntuinisticLanguageModel(64)
logits, loss = ilmodel(x,y)
print(logits)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
gen = ilmodel.generate(idx, max_new_tokens=33*3-1)[0].tolist()
print("gen", gen)
out = manager.format_output(gen)
print("A", out)
print("B", "".join([x for x in detokenizer(out) if x is not None]))


