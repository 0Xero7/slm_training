import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalMHA(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        print("Using classical attention")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):        
        B, T, D = x.size()  # batch_size, sequence_length, d_model
        H = self.num_heads
        
        # Project and reshape
        q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
        k = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
        v = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        scores = scores * self.scale
        
        # Optional: Causal mask for decoder self-attention
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.view(1, 1, T, T), float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        output = torch.matmul(attn, v)  # (B, H, T, head_dim)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(output)
        
        return output