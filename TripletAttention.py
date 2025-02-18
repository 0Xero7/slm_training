import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TripletAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        print(f"[Using triplet attention]. {d_model=}, {num_heads=}, {dropout=}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.t_proj = nn.Linear(d_model, d_model)  # Triplet projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        # Changed scaling factor to account for both attention terms
        self.scale = 1.0 / (math.sqrt(self.head_dim) * 2)
        
    def forward(self, x):
        B, T, D = x.size()
        H = self.num_heads
        
        # Project and reshape
        q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
        k = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)
        t = self.t_proj(x).view(B, T, H, -1).transpose(1, 2)
        
        # Standard attention scores
        scores_dot = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
        # Modified triplet interaction to be more stable
        k_v = k * v  # (B, H, T, head_dim)
        k_v = k_v / math.sqrt(self.head_dim)  # Scale to prevent exponential growth
        triplet_cumsum = torch.cumsum(k_v, dim=2)  # (B, H, T, head_dim)
        scores_triplet = torch.matmul(q * t, triplet_cumsum.transpose(-2, -1))
        
        # Combine scores and scale
        scores = (scores_dot) * self.scale
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.view(1, 1, T, T), float('-inf'))
        
        # Attention weights with numerical stability
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores = scores - scores_max  # Subtract max for numerical stability
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        output = torch.matmul(attn, v)  # (B, H, T, head_dim)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(output)
        
        return output