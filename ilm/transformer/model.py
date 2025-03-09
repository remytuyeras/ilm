from typing import Optional, Callable
from tqdm import trange
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

def format_context(raw_text: str, tokenizer: Callable[[str],list[str]]) -> torch.Tensor:
    # return a single_context of dimension (T, )
    return torch.tensor(
        [int(y) for x in tokenizer(raw_text) for y in x.split(":")], 
        dtype=torch.long
        )

def gather_tokens(tokens: list[int], syllable_num: int = 3) -> list[str]:
    output = []

    # prepare the parsing of (syllable_num)-tokens
    quotient = len(tokens) // syllable_num
    remainder = len(tokens) % syllable_num
    
    # convert single_context into list of (syllable_num)-tokens
    for i in range(quotient):
        output.append(":".join([str(tokens[syllable_num*i+j]) for j in range(syllable_num)]))
    
    # handle potentially incomplete (syllable_num)-tokens
    if remainder > 0:
        output.append(":".join([str(tokens[x]) for x in range(syllable_num * quotient, len(tokens))] + ["?"] * remainder))
    
    return output

class TrainingManager(object):
    
    def __init__(self, raw_text, tokenizer, device=torch.device("cpu")):
        
        # raw = raw_text.split("\n")
        # random.shuffle(raw)
        # raw_text = "\n\n".join(raw)
        
        # Batch dimension: B
        self.batch_size = 32
        # Time dimension: T
        self.block_size = 3 * 8

        self.device = device

        # Encoder for tokens
        self.tokenizer = tokenizer
        # Process input with token encoder
        self.text_tokens = self.format_input(raw_text)
        
        # Create training and validation test
        self.n_dataset = len(self.text_tokens)
        print("n_dataset =", self.n_dataset)
        self.n_training = int(0.8 * self.n_dataset)
        print("n_training =",self.n_training)
        
        # Training data
        self.training_data = self.text_tokens[:self.n_training].to(self.device)
        # Validation data
        self.validation_data = self.text_tokens[self.n_training:].to(self.device)
                
    def get_batch(self, split):
        data = self.training_data if split == "train" else self.validation_data

        # random indices accross document
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # size (B, T) = Batch dimension, Time dimension
        x = torch.stack([data[i:i+self.block_size] for i in indices])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in indices])

        return x, y
    
    def format_output(self, tokens: list[int], syllable_num: int = 3) -> list[str]:
        return gather_tokens(tokens=tokens, syllable_num=syllable_num)
    
    def format_input(self, raw_text: str) -> torch.Tensor:
        return format_context(raw_text=raw_text, tokenizer=self.tokenizer)

       
class IntuinisticLanguageModel(nn.Module):
    
    def __init__(self, vocab_size=64, device=torch.device("cpu")):
        super().__init__()
        self.device=device
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size).to(self.device)

        # print(self.token_embedding_table(torch.tensor([0,1], dtype=torch.long)))

    def forward(self, batched_context: torch.Tensor , batched_targets: Optional[torch.Tensor] = None):
        # batched_context: (B, T) -> batched_logits: (B, T, C)
        batched_logits: torch.Tensor = self.token_embedding_table(batched_context)
        
        if batched_targets is None:
            loss = None
        
        else:
            B, T, C = batched_logits.shape # Batch, Time, Channel
            # concatenate logits C-vectors ovar all B-batches
            flattened_logits = batched_logits.view(B * T, C).to(self.device)
            # concatenate token T-tuples over all B-batches
            flattened_targets = batched_targets.view(B * T).to(self.device)
            # compute cross_entropy loss where each entry of flattened_targets becomes the binary C-vector
            loss = F.cross_entropy(flattened_logits, flattened_targets)
        
        return batched_logits, loss
    
    def train_model(self, manager: TrainingManager, epoch_num: int = 1000):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        progress_bar = trange(epoch_num, desc="Training")
        
        for _ in progress_bar:
            x, y = manager.get_batch("train")
            logits, loss = self(x, y); loss: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    def save_model(self, model_path = "iml_model.pth"):
        torch.save(self.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

    def load_model(self, model_path="iml_model.pth"):
        self.load_state_dict(torch.load(model_path))
        self.eval().to(self.device)
        print(f"Model weights loaded from {model_path}")

    def generate(self, batched_context: torch.Tensor, max_new_tokens: int, temperature: float = 0): 
        '''
        Understood
        '''
        batched_context_ = batched_context.to(self.device)
        progress_bar = trange(max_new_tokens, desc="Inference")
        # single_context should be (B, T) where T grows
        for _ in progress_bar:
            # give batched logits and loss
            batched_logits, loss = self(batched_context_)
            # get the logits (prediction) for the last context token
            last_logits = batched_logits[:,-1,:]
   
            # compute associated probability distribution(s) (used in cross entropy)
            if temperature == 0:
                # Deterministic: choose the highest probability token
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # Scale logits by the temperature.
                #   - Lower temperatures (<1) make the softmax distribution sharper (more peaked),
                #     so the highest logit dominates.
                #   - Higher temperatures (>1) flatten the distribution, making the sampling more random.
                scaled_logits = last_logits / temperature

                # Compute the probability distribution over tokens.
                scaled_probs = F.softmax(scaled_logits, dim=-1)

                # Use torch.multinomial to sample the next token from the probability distribution.
                # IMPORTANT:
                # torch.multinomial is stochastic; it randomly draws an index based on the probabilities in 'scaled_probs'.
                # This means that even if one token has the highest probability, it might not always be selected.
                # If you need deterministic behavior (always selecting the token with the highest probability),
                # you would use torch.argmax instead.
                next_token = torch.multinomial(scaled_probs, num_samples=1)

            # extend the context to (B, T+1)
            batched_context_ = torch.cat((batched_context_, next_token), dim=1)

        return batched_context_