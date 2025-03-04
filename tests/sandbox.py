import sys
sys.path.insert(1, "./")
import ilm

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1234)

from tqdm import trange

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
ilmodel = ilm.IntuinisticLanguageModel(64)

x, y = manager.get_batch("train")
logits, loss = ilmodel(x,y)
print(logits)
print(loss)

optimizer = torch.optim.AdamW(ilmodel.parameters(), lr=1e-3)

manager.batch_size = 32
manager.block_size = 20 * 3

num_steps = 10000
progress_bar = trange(num_steps, desc="Training")
for step in progress_bar:
    x, y = manager.get_batch("train")
    logits, loss = ilmodel(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss: torch.Tensor
    loss.backward()
    optimizer.step()
    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

# idx = torch.zeros((1,1), dtype=torch.long)
idx = manager.format_input("Could you please").unsqueeze(0)
print(idx)
gen = ilmodel.generate(idx, max_new_tokens=330*3)[0].tolist()
print("gen", gen)
out = manager.format_output(gen)
print("A", out)
print("B", "".join([x for x in detokenizer(out) if x is not None]))


