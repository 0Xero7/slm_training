import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SPSTPAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pre_stp_norm = nn.LayerNorm(3)  # For subspace projections
        
        # Subspace projection matrices (for k and v)
        self.proj_k = nn.Linear(self.head_dim, 3)  # Project to 3D subspace
        self.proj_v = nn.Linear(self.head_dim, 3)
        self.proj_q_stp = nn.Linear(self.head_dim, 3)  # NEW: Match query dim

        # Add learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))  # NEW
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        H, HD = self.num_heads, self.head_dim
        
        # Project queries/keys/values
        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)  # [B, H, T, HD]
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)

        # Add nuclear validation
        assert not torch.isnan(q).any(), "NaN in queries"
        assert not torch.isinf(k).any(), "Inf in keys"
        # # Project k/v to 3D subspaces
        # k_sub = self.proj_k(k)  # [B, H, T, 3]
        # v_sub = self.proj_v(v)  # [B, H, T, 3]
        
        # # Compute STP in subspace
        # cross_kv = torch.cross(k_sub, v_sub, dim=-1)  # [B, H, T, 3]

        # In forward():
        
        # Project queries for STP
        q_stp = self.proj_q_stp(q)  # [B,H,T,3] NEW
        k_sub = self.proj_k(k)
        v_sub = self.proj_v(v)
        k_sub = self.pre_stp_norm(k_sub)  # Stabilize magnitudes
        v_sub = self.pre_stp_norm(v_sub)

        
        # Compute STP scores
        cross_kv = torch.cross(
            self.proj_k(k), 
            self.proj_v(v),
            dim=-1
        )
        stp_scores = torch.matmul(q_stp, cross_kv.transpose(-2, -1))  # FIXED
        
        # Balanced combination
        dot_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]
        scores = (dot_scores + stp_scores) * self.temperature  # CHANGED


        # cross_kv = torch.cross(k_sub, v_sub, dim=-1)
        # # print("Cross product range:", cross_kv.min().item(), cross_kv.max().item())  # Should be sane
        # stp_scores = torch.matmul(q[..., :3], cross_kv.transpose(-2, -1))  # [B, H, T, T]
        # # print("STP scores mean:", stp_scores.mean().item())  # Should be ~0
        
        # # Standard attention scores
        
        # # Combine scores
        # scores = (dot_scores + stp_scores) * self.scale
        # print("Combined scores std:", scores.std().item())  # Should be <100
        attn = F.softmax(scores, dim=-1)
        
        # Output
        output = torch.matmul(attn, v)
        return self.out_proj(output.transpose(1, 2).contiguous().view(B, T, D))

class STPAttention_Bad(nn.Module):
    def __init__(self, d_model, num_heads=40, dropout=0.1, is_causal=False):
        super().__init__()
        print(f"[Using STP Attention] d_model={d_model}, num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert self.head_dim == 3, "STP requires head_dim=3 (d_model must be 3*num_heads)"

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, D = x.size()
        H, HD = self.num_heads, self.head_dim

        # Project and reshape
        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)

        # Standard attention scores
        scores_dot = torch.matmul(q, k.transpose(-2, -1))
        
        # Scalar triple product scores
        cross_kv = torch.cross(k, v, dim=-1)
        scores_stp = torch.matmul(q, cross_kv.transpose(-2, -1))

        # Combine and scale
        scores = (scores_dot + scores_stp) * self.scale

        # Causal masking
        if self.is_causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

        # Attention computation
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Context aggregation
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_proj(context)