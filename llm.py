import torch
import torch.nn as nn
from torch.nn import functional as F

import config


# get batch from data
def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x, y


class SelfAttentionHead(nn.Module):
    def __init__(self, num_embed_dim, head_size, block_size, dropout):
        super().__init__()
        self.register_buffer('low_triangular', torch.tril(torch.ones(block_size, block_size)))

        self.key = nn.Linear(num_embed_dim, head_size, bias=False)
        self.query = nn.Linear(num_embed_dim, head_size, bias=False)
        self.value = nn.Linear(num_embed_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch, time step, channels
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) 

        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 

        # fill zeros with negative infinity
        weights = weights.masked_fill(self.low_triangular[:T, :T] == 0, float('-inf')) 

        # normalize values to probability distribution
        weights = F.softmax(weights, dim=-1) 

        # randomly dropout nodes
        weights = self.dropout(weights)
        
        # matrix multiply the weights by the value
        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_embed_dim, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(num_embed_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, num_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        out = self.proj(out)

        # randomly dropout nodes
        out = self.dropout(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, num_embed_dim, num_head, dropout):
        super().__init__()
        head_size = num_embed_dim // num_head
        self.self_attention = MultiHeadAttention(num_embed_dim, num_head, head_size, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(num_embed_dim, 4 * num_embed_dim),
            nn.ReLU(),
            nn.Linear(4 * num_embed_dim, num_embed_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm_1 = nn.LayerNorm(num_embed_dim)
        self.layer_norm_2 = nn.LayerNorm(num_embed_dim)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feedforward(self.layer_norm_2(x))
        return x
    

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, num_embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, num_embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(num_embed_dim, n_head=num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(num_embed_dim) 
        self.head = nn.Linear(num_embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding 
        x = self.blocks(x) 
        x = self.layer_norm(x) 
        logits = self.head(x)

        # if target is not specified, loss is not required
        if targets is None:
            return logits, None
      
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]

            # get the predictions
            logits, _ = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] 

            # get probabilities
            probs = F.softmax(logits, dim=-1) 

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def train(self, data):
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]

        optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

        for iteration in range(config.num_iterations):
            # get batch of training data
            x, y = get_batch(train_data)

            # evaluate loss for x, y
            _, loss = self(x, y)

            # every config.evaluation_interval, determine loss on validation set
            if iteration % config.evaluation_interval == 0 or iteration == config.num_iterations - 1:
                with torch.no_grad():
                    # set model to evaluation mode
                    self.eval()

                    losses = {}
                    for split in ['train', 'val']:
                        l = torch.zeros(config.evaluation_interval)
                        for k in range(config.evaluation_interval):
                            X, Y = get_batch(train_data if split == 'train' else val_data)
                            _, loss = self(X, Y)
                            l[k] = loss.item()
                        losses[split] = losses.mean()
                    
                    # set model to train mode
                    self.train()
                    print(f"Step {iteration}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

            # set gradients to zero
            optimizer.zero_grad()

            # calculate gradients
            loss.backward()

            # update parameters based on gradients
            optimizer.step()
