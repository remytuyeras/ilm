from typing import Optional, Callable, List
from tqdm import trange
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Use the same seed before training and testing

def format_context(raw_text: str, tokenizer: Callable[[str], List[str]]) -> torch.Tensor:
    # return a single_context of dimension (T, )
    return torch.tensor(
        [int(y) for x in tokenizer(raw_text) for y in x.split(":")], 
        dtype=torch.long
        )

def gather_tokens(tokens: List[int], syllable_num: int = 3) -> List[str]:
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
        
        # raw = raw_text.split("\n\n")
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
    
    def format_output(self, tokens: List[int], syllable_num: int = 3) -> List[str]:
        return gather_tokens(tokens=tokens, syllable_num=syllable_num)
    
    def format_input(self, raw_text: str) -> torch.Tensor:
        return format_context(raw_text=raw_text, tokenizer=self.tokenizer)


class ILMHead(nn.Module):
    
    def __init__(self,
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 device=torch.device("cpu"),
                 dropout=0.2):
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.device = device

        self.key = nn.Linear(embedding_dim, head_size, bias=False).to(self.device)
        self.query = nn.Linear(embedding_dim,head_size, bias=False).to(self.device)
        self.value = nn.Linear(embedding_dim, head_size, bias=False).to(self.device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(self.device))

        self.dropout = nn.Dropout(dropout)

    def forward(self, batched_context: torch.Tensor):
        T = batched_context.shape[1]

        q: torch.Tensor = self.query(batched_context) # (B, T, H)
        k: torch.Tensor = self.key(batched_context) # (B, T, H)
        v: torch.Tensor = self.value(batched_context) # (B, T, H)

        weights: torch.Tensor = q @ k.transpose(-2,-1) / (self.head_size ** 0.5) # (B, T, T)
        
        weights = weights.masked_fill(
            self.tril[:T,:T] == 0, 
            float("-inf")) # Decoder structure
        
        weights = F.softmax(weights, dim=-1) # (B, T)
        weights = self.dropout(weights)
        output = weights @ v # (B, H)

        return output

class ILMMultiHead(nn.Module):
    
    def __init__(self,
                 head_num=4, 
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 device=torch.device("cpu"),
                 dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([ILMHead(embedding_dim, block_size, head_size, device, dropout) for _ in range(head_num)]) 
        self.proj = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.dropout = nn.Dropout(dropout)
       
    def forward(self,input_emb):
        emb = torch.cat([ h(input_emb)for h in self.heads], dim=-1)
        emb = self.proj(emb)
        emb = self.dropout(emb)
        return emb


class ILMFeedForward(nn.Module):
    
    def __init__(self, embedding_dim=32, device=torch.device("cpu"), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
            ).to(device)
        
    def forward(self,batched_embeddings):
        return self.net(batched_embeddings)

class ILMBlock(nn.Module):
    
    def __init__(self, 
                 head_num=4,
                 embedding_dim=32, 
                 block_size=12,
                 device=torch.device("cpu"),
                 dropout=0.2):
        super().__init__()
        head_size = embedding_dim // head_num
        self.sa_heads = ILMMultiHead(head_num=head_num,
                                     embedding_dim=embedding_dim, 
                                     block_size=block_size,
                                     head_size=head_size,
                                     device=device,
                                     dropout=dropout)
        self.ffwrd = ILMFeedForward(embedding_dim=embedding_dim, device=device, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_dim).to(device)
        self.ln2 = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwrd(self.ln2(x))
        return x

class IntuinisticLanguageModel(nn.Module):
    
    def __init__(self, 
                 vocab_size=64, # vocabulary
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 layer_num=4,
                 device=torch.device("cpu"),
                 dropout=0.2):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.head_size = head_size

        self.device=device
        
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.pos_embedding_table = nn.Embedding(block_size, embedding_dim).to(self.device)

        # self.sa_head = ILMHead(
        #     embedding_dim=embedding_dim,
        #     block_size=block_size,
        #     head_size=embedding_dim,
        #     device=device)

        # self.sa_heads = ILMMultiHead(
        #     head_num=4,
        #     embedding_dim=embedding_dim,
        #     block_size=block_size,
        #     head_size=embedding_dim//4,
        #     device=device)
        
        # self.ffwrd = ILMFeedForward(embedding_dim=embedding_dim, device=device)

        self.blocks = nn.Sequential(
            *[ILMBlock(
                head_num=4,
                embedding_dim=embedding_dim,
                block_size=block_size,
                device=device,
                dropout=dropout)
            for _ in range(layer_num)])
        
        self.ln_f = nn.LayerNorm(embedding_dim).to(device)

        self.lm_head = nn.Linear(embedding_dim, vocab_size).to(self.device)

        # print(self.token_embedding_table(torch.tensor([0,1], dtype=torch.long)))

    def forward(self, batched_context: torch.Tensor , batched_targets: Optional[torch.Tensor] = None):
        # self.eval()
        
        T = batched_context.shape[1]
        # batched_context: (B, T) -> batched_tok_emb: (B, T, C)
        batched_tok_emb: torch.Tensor = self.token_embedding_table(batched_context)
        # position vectors  -> batched_tok_emb: (T, C)
        batched_pos_emb: torch.Tensor = self.pos_embedding_table(torch.arange(T, device=self.device).to(self.device))
        # embedding and postiond 
        batched_emb = batched_tok_emb + batched_pos_emb
        # self attention
        batched_emb = self.blocks(batched_emb)
        batched_emb = self.ln_f(batched_emb)
        # batched_embeddings: (B, T, C) -> batched_logits: (B, T, V) ; V=vocabulary
        batched_logits: torch.Tensor = self.lm_head(batched_emb)
        
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
    

    @torch.no_grad()
    def estimate_loss(self, manager: TrainingManager):
        eval_iters = 100
        out = {}
        self.eval()
        for split in ["train", "validate"]:
            losses = torch.zeros(eval_iters).to(self.device)
            for k in range(eval_iters):
                x, y = manager.get_batch(split)
                _, loss = self(x,y); loss: torch.Tensor
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out


    def train_model(self, manager: TrainingManager, epoch_num: int = 5000, lr: float =1e-3):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        progress_bar = trange(epoch_num, desc="Training")

        losses = self.estimate_loss(manager=manager)
        loss_msg = f"loss_trn:{losses['train']:.4f}, loss_val:{losses['validate']:.4f}"
        progress_bar.set_postfix(loss=loss_msg)
        
        for bar_step in progress_bar:
            self.train()
            x, y = manager.get_batch("train")
            logits, loss = self(x, y); loss: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if bar_step % 500 == 0:
                losses = self.estimate_loss(manager=manager)
                loss_msg = f"loss_trn:{losses['train']:.4f}, loss_val:{losses['validate']:.4f}"
                progress_bar.set_postfix(loss=loss_msg)
            

    def save_model(self, model_path = "iml_model.pth"):
        for param in self.parameters():
            param.data = param.data.to(torch.float32)
        torch.save(self.state_dict(), model_path, _use_new_zipfile_serialization=False)
        print(f"Model weights saved to {model_path}")


    def load_model(self, model_path="iml_model.pth"):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(torch.float32)  # Ensure all parameters remain float32
        self.to(self.device).eval()
        print(f"Model weights loaded from {model_path}")


    def generate(self, batched_context: torch.Tensor, max_new_tokens: int, temperature: float = 0, top_k: Optional[int] = None): 
        '''
        Understood
        '''
        self.eval()
        batched_context_ = batched_context.to(self.device)
        progress_bar = trange(max_new_tokens, desc="Inference")
        # single_context should be (B, T) where T grows
        for _ in progress_bar:
            # give batched logits and loss
            batched_logits, loss = self(batched_context_[: , -self.block_size:])
            # get the logits (prediction) for the last context token
            last_logits = batched_logits[:,-1,:]
   
            # compute associated probability distribution(s) (used in cross entropy)
            if temperature == 0:
                # Deterministic: choose the highest probability token
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # Scale logits by the temperature if non-zero.
                #   - Lower temperatures (<1) make the softmax distribution sharper (more peaked),
                #     so the highest logit dominates.
                #   - Higher temperatures (>1) flatten the distribution, making the sampling more random.
                scaled_logits = last_logits / temperature

                if top_k is not None:
                    top_k = min(top_k, scaled_logits.shape[-1])
                    top_values, _ = torch.topk(scaled_logits, k=top_k)
                    threshold = top_values[:, -1].unsqueeze(-1)
                    scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))

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
