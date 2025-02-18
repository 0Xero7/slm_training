import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# import EnchancedTextDataset
import TripletAttention as ta
import ClassicalMHA as cm
import STPAttention as sa

# ---------------------------
# Model Components with Dropout
# ---------------------------
# class CausalTripletAttention(nn.Module):
#     def __init__(self, d_model, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         # Linear projections for queries, keys, values.
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         """
#         x: (B, T, d_model)
#         Returns:
#             output: (B, T, d_model)
#         """
#         B, T, D = x.size()
#         q = self.q_proj(x)  # (B, T, D)
#         k = self.k_proj(x)  # (B, T, D)
#         v = self.v_proj(x)  # (B, T, D)
        
#         # Compute standard scaled dot-product attention scores.
#         scores_dot = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, T, T)
        
#         # Compute an extra "triplet" term using elementwise multiplication.
#         extra = torch.einsum('bid,bjd,bjd->bij', q, k, v)  # (B, T, T)
        
#         # Total attention score.
#         scores = scores_dot + extra
        
#         # Create a causal mask: allow each position to attend only to itself and previous positions.
#         mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
#         scores.masked_fill_(mask, float('-inf'))
        
#         # Softmax over the key dimension.
#         attn = F.softmax(scores, dim=-1)  # (B, T, T)
#         attn = self.dropout(attn)
        
#         # Weighted sum over values.
#         output = torch.matmul(attn, v)  # (B, T, D)
#         return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, dropout=0.3, use_experimental=True):
        super().__init__()
        if use_experimental:
            # self.attn = ta.TripletAttention(d_model, dropout=dropout)
            self.attn = sa.SPSTPAttention(d_model, dropout=dropout)
        else:
            self.attn = cm.ClassicalMHA(d_model, dropout=dropout) 
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection.
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward network with residual connection.
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, ff_hidden=512, max_seq_len=256, dropout=0.1, use_experimental=True):
        super().__init__()
        # Increase dropout significantly
        self.emb_dropout = nn.Dropout(0.3)  # New embedding dropout
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, ff_hidden, dropout=dropout, use_experimental=use_experimental) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)  # Add final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

        # Weight tying between embedding and output layer
        self.token_embed.weight = self.lm_head.weight
        
    def forward(self, input_ids):
        """
        input_ids: (B, T)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()
        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.emb_dropout(x)  # Apply dropout to embeddings
        # Add learned positional embeddings.
        x = x + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)  # Final layer norm
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        Autoregressively generate tokens given a prompt.
        input_ids: (B, T)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Ensure we don't exceed max_seq_len.
            x_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits = self.forward(x_cond)
            # Get logits for the last token.
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids