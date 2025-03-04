from typing import Optional   
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

import random


class TrainingManager(object):
    
    def __init__(self, raw_text, tokenizer):
        
        # raw = raw_text.split("\n")
        # random.shuffle(raw)
        # raw_text = "\n\n".join(raw)
        
        self.block_size = 8
        self.batch_size = 4

        self.tokenizer = tokenizer
        self.text_tokens = self.format_input(raw_text)
        
        self.n_dataset = len(self.text_tokens)
        print("n_dataset =", self.n_dataset)
        self.n_training = int(0.8*self.n_dataset)
        print("n_training =",self.n_training)
        
        self.training_data = self.text_tokens[:self.n_training]
        self.validation_data = self.text_tokens[self.n_training:]
                
    def get_batch(self, split):
        data = self.training_data if split == "train" else self.validation_data
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in idx])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in idx])
        return x, y
    
    def format_output(self, tokens, syllable_num = 3):
        output = []
        quotient = len(tokens) // syllable_num
        remainder = len(tokens) % syllable_num
        for i in range(quotient):
            output.append(":".join([str(tokens[syllable_num*i+j]) for j in range(syllable_num)]))
        if remainder > 0:
            output.append(":".join([str(x) for x in range(syllable_num * quotient, len(tokens))] + ["?"] * remainder))
        return output
    
    def format_input(self, raw_text):
        return torch.tensor(
            [int(y) for x in self.tokenizer(raw_text) for y in x.split(":")], 
            dtype=torch.long
            )
        
        
class IntuinisticLanguageModel(nn.Module):
    
    def __init__(self, vocab_size=64):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor , targets: Optional[torch.Tensor] = None ):
        # print("ids", idx)
        # print("targets", targets)

        logits: torch.Tensor = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print(f"B={B}, T={T}, C={C}")
            logits = logits.view(B * T, C)
            # print("logits", logits)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            # print("loss", loss)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx