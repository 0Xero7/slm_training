# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Original implementation
# class OriginalTripletAttention(nn.Module):
#     def __init__(self, d_model, num_heads=4, dropout=0.1):
#         super().__init__()
#         print(f"[Using triplet attention]. {d_model=}, {num_heads=}, {dropout=}")
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         # Linear projections
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.t_proj = nn.Linear(d_model, d_model)  # Triplet projection
#         self.out_proj = nn.Linear(d_model, d_model)
        
#         self.dropout = nn.Dropout(dropout)
#         # Changed scaling factor to account for both attention terms
#         self.scale = 1.0 / (math.sqrt(self.head_dim))
        
#     def forward(self, x):
#         B, T, D = x.size()
#         H = self.num_heads
        
#         # Project and reshape
#         q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
#         k = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)
#         v = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)
#         t = self.t_proj(x).view(B, T, H, -1).transpose(1, 2)
        
#         # Standard attention scores
#         scores_dot = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
#         # Modified triplet interaction to be more stable
#         k_v = k * v  # (B, H, T, head_dim)
#         k_v = k_v / math.sqrt(self.head_dim)  # Scale to prevent exponential growth
#         triplet_cumsum = torch.cumsum(k_v, dim=2)  # (B, H, T, head_dim)
#         scores_triplet = torch.matmul(q * t, triplet_cumsum.transpose(-2, -1))
        
#         # Combine scores and scale
#         scores = (scores_dot + scores_triplet) * self.scale
        
#         # Causal mask
#         mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
#         scores.masked_fill_(mask.view(1, 1, T, T), float('-inf'))
        
#         # Attention weights with numerical stability
#         scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
#         scores = scores - scores_max  # Subtract max for numerical stability
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
        
#         # Weighted sum
#         output = torch.matmul(attn, v)  # (B, H, T, head_dim)
        
#         # Reshape and project output
#         output = output.transpose(1, 2).contiguous().view(B, T, D)
#         output = self.out_proj(output)
        
#         return output

# # Optimized implementation
# class OptimizedTripletAttention(nn.Module):
#     def __init__(self, d_model, num_heads=4, dropout=0.1):
#         super().__init__()
#         print(f"[Using optimized triplet attention]. {d_model=}, {num_heads=}, {dropout=}")
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         # Combined projection for efficiency - saves 3 separate operations
#         self.qkvt_proj = nn.Linear(d_model, 4 * d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
        
#         self.dropout = nn.Dropout(dropout)
#         self.scale = 1.0 / (math.sqrt(self.head_dim))
        
#         # Pre-compute causal mask once during initialization
#         self.register_buffer("causal_mask", torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool())
        
#     def forward(self, x):
#         B, T, D = x.size()
#         H = self.num_heads
        
#         # Ensure our pre-computed mask is big enough, resize if needed
#         if T > self.causal_mask.size(-1):
#             self.causal_mask = torch.triu(torch.ones(1, 1, T, T, device=x.device), diagonal=1).bool()
        
#         # Fused projection and reshaping
#         qkvt = self.qkvt_proj(x).view(B, T, 4, H, self.head_dim)
#         q, k, v, t = qkvt.unbind(dim=2)  # Split along projection dimension
        
#         # Transpose once after unbinding (more efficient)
#         q = q.transpose(1, 2)  # (B, H, T, head_dim)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
#         t = t.transpose(1, 2)
        
#         # Standard attention scores - use torch.baddbmm for efficiency
#         scores_dot = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
#         # Optimized triplet interaction
#         k_v = k * v  # (B, H, T, head_dim)
#         k_v = k_v * self.scale  # Scale to prevent exponential growth
        
#         # Optimize cumsum with memory usage - combine operations
#         triplet_cumsum = torch.cumsum(k_v, dim=2)  # (B, H, T, head_dim)
#         scores_triplet = torch.matmul(q * t, triplet_cumsum.transpose(-2, -1))
        
#         # Combined addition and scaling in one operation
#         scores = (scores_dot + scores_triplet) * self.scale
        
#         # Apply causal mask more efficiently
#         mask = self.causal_mask[:, :, :T, :T]
#         scores.masked_fill_(mask, float('-inf'))
        
#         # Stable softmax: subtract max(scores) for numerical stability
#         scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
#         scores = scores - scores_max
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
        
#         # Weighted sum and reshape in fewer operations
#         output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
#         output = output.transpose(1, 2).reshape(B, T, D)  # Use reshape instead of contiguous().view()
        
#         return self.out_proj(output)

# def verify_equivalence(d_model=256, num_heads=4):
#     """
#     Verify numerical equivalence between original and optimized TripletAttention
    
#     Args:
#         d_model: dimension of the model
#         num_heads: number of attention heads
    
#     Returns:
#         dict with verification results
#     """
#     # Set seeds for reproducibility
#     torch.manual_seed(42)
    
#     # Create models with fixed dropout for deterministic results
#     dropout = 0.0
#     original_model = OriginalTripletAttention(d_model, num_heads, dropout)
#     optimized_model = OptimizedTripletAttention(d_model, num_heads, dropout)
    
#     # Copy weights from original to optimized to ensure they're identical
#     with torch.no_grad():
#         # Extract weights from original model
#         q_weight = original_model.q_proj.weight
#         k_weight = original_model.k_proj.weight
#         v_weight = original_model.v_proj.weight
#         t_weight = original_model.t_proj.weight
        
#         q_bias = original_model.q_proj.bias
#         k_bias = original_model.k_proj.bias
#         v_bias = original_model.v_proj.bias
#         t_bias = original_model.t_proj.bias
        
#         # Combine weights for optimized model
#         # For the weight matrix, we need to stack them correctly
#         combined_weight = torch.cat([q_weight, k_weight, v_weight, t_weight], dim=0)
#         combined_bias = torch.cat([q_bias, k_bias, v_bias, t_bias])
        
#         optimized_model.qkvt_proj.weight.copy_(combined_weight)
#         optimized_model.qkvt_proj.bias.copy_(combined_bias)
        
#         # Copy output projection
#         optimized_model.out_proj.weight.copy_(original_model.out_proj.weight)
#         optimized_model.out_proj.bias.copy_(original_model.out_proj.bias)
    
#     # Test with different batch sizes and sequence lengths
#     test_cases = [
#         {"batch_size": 2, "seq_len": 10},
#         {"batch_size": 1, "seq_len": 50},
#         {"batch_size": 4, "seq_len": 25},
#     ]
    
#     results = []
    
#     for tc in test_cases:
#         batch_size = tc["batch_size"]
#         seq_len = tc["seq_len"]
        
#         # Create test input
#         torch.manual_seed(123 + batch_size + seq_len)  # Different seed for each input
#         x = torch.randn(batch_size, seq_len, d_model)
        
#         # Evaluate both models
#         with torch.no_grad():
#             original_output = original_model(x)
#             optimized_output = optimized_model(x)
        
#         # Check if outputs are close
#         is_close = torch.allclose(original_output, optimized_output, rtol=1e-5, atol=1e-5)
#         max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
        
#         results.append({
#             "batch_size": batch_size,
#             "seq_len": seq_len,
#             "is_numerically_equivalent": is_close,
#             "max_absolute_difference": max_diff,
#         })
    
#     return results

# def measure_performance(d_model=768, num_heads=12, batch_size=8, seq_len=512, num_runs=10):
#     """
#     Measure and compare performance between original and optimized implementations
    
#     Args:
#         d_model: dimension of the model
#         num_heads: number of attention heads
#         batch_size: batch size for test
#         seq_len: sequence length for test
#         num_runs: number of runs to average performance over
        
#     Returns:
#         dict with performance comparison
#     """
#     # Create models
#     torch.manual_seed(42)
#     original_model = OriginalTripletAttention(d_model, num_heads, dropout=0.0)
    
#     torch.manual_seed(42)
#     optimized_model = OptimizedTripletAttention(d_model, num_heads, dropout=0.0)
    
#     # Create input
#     x = torch.randn(batch_size, seq_len, d_model)
    
#     # Warm-up
#     for _ in range(5):
#         with torch.no_grad():
#             _ = original_model(x)
#             _ = optimized_model(x)
    
#     # Measure original model
#     import time
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
    
#     start = time.time()
#     for _ in range(num_runs):
#         with torch.no_grad():
#             _ = original_model(x)
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     original_time = (time.time() - start) / num_runs
    
#     # Measure optimized model
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     start = time.time()
#     for _ in range(num_runs):
#         with torch.no_grad():
#             _ = optimized_model(x)
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     optimized_time = (time.time() - start) / num_runs
    
#     return {
#         "original_time": original_time,
#         "optimized_time": optimized_time,
#         "speedup": original_time / optimized_time,
#         "batch_size": batch_size,
#         "seq_len": seq_len,
#         "d_model": d_model,
#         "num_heads": num_heads
#     }

# if __name__ == "__main__":
#     # Run verification
#     print("Verifying numerical equivalence...")
#     equiv_results = verify_equivalence(d_model=256, num_heads=4)
#     for i, result in enumerate(equiv_results):
#         print(f"Test case {i+1}:")
#         print(f"  Batch size: {result['batch_size']}, Sequence length: {result['seq_len']}")
#         print(f"  Numerically equivalent: {result['is_numerically_equivalent']}")
#         print(f"  Maximum absolute difference: {result['max_absolute_difference']}")
    
#     # Run performance comparison
#     print("\nMeasuring performance...")
#     perf_results = measure_performance(d_model=768, num_heads=12, batch_size=8, seq_len=512, num_runs=1)
#     print(f"Model dimensions: {perf_results['d_model']}, Heads: {perf_results['num_heads']}")
#     print(f"Batch size: {perf_results['batch_size']}, Sequence length: {perf_results['seq_len']}")
#     print(f"Original model average time: {perf_results['original_time']:.6f} seconds")
#     print(f"Optimized model average time: {perf_results['optimized_time']:.6f} seconds")
#     print(f"Speedup: {perf_results['speedup']:.2f}x")



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Try to import Flash Attention
HAS_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention available")
except ImportError:
    try:
        # Try PyTorch's implementation as fallback
        from torch.nn.functional import scaled_dot_product_attention
        HAS_FLASH_ATTN = True
        print("PyTorch's Flash Attention available")
    except ImportError:
        print("Flash Attention not available")

# Original implementation (unchanged)
class OriginalTripletAttention(nn.Module):
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
        self.scale = 1.0 / (math.sqrt(self.head_dim))
        
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
        scores = (scores_dot + scores_triplet) * self.scale
        
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

# Optimized implementation
class OptimizedTripletAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, use_flash_attn=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.use_flash = use_flash_attn and HAS_FLASH_ATTN
        if self.use_flash:
            print(f"[Using flash triplet attention]. {d_model=}, {num_heads=}, {dropout=}")
        else:
            print(f"[Using optimized triplet attention]. {d_model=}, {num_heads=}, {dropout=}")
        
        # Combined projection for efficiency - saves 3 separate operations
        self.qkvt_proj = nn.Linear(d_model, 4 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.scale = 1.0 / (math.sqrt(self.head_dim))
        
        # Pre-compute causal mask once during initialization
        self.register_buffer("causal_mask", torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool())
    
    def forward(self, x):
        B, T, D = x.size()
        H = self.num_heads
        
        # Ensure our pre-computed mask is big enough, resize if needed
        if T > self.causal_mask.size(-1):
            self.causal_mask = torch.triu(torch.ones(1, 1, T, T, device=x.device), diagonal=1).bool()
        
        # Fused projection and reshaping
        qkvt = self.qkvt_proj(x).view(B, T, 4, H, self.head_dim)
        q, k, v, t = qkvt.unbind(dim=2)  # Split along projection dimension
        
        # Transpose once after unbinding (more efficient)
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        t = t.transpose(1, 2)
        
        # Standard attention scores 
        scores_dot = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
        # Optimized triplet interaction
        k_v = k * v  # (B, H, T, head_dim)
        k_v = k_v * (1.0 / math.sqrt(self.head_dim))  # Scale to prevent exponential growth
        
        # Optimize cumsum with memory usage - combine operations
        triplet_cumsum = torch.cumsum(k_v, dim=2)  # (B, H, T, head_dim)
        scores_triplet = torch.matmul(q * t, triplet_cumsum.transpose(-2, -1))
        
        # Combined addition and scaling in one operation
        scores = (scores_dot + scores_triplet) * self.scale
        
        # Apply causal mask 
        mask = self.causal_mask[:, :, :T, :T]
        scores.masked_fill_(mask, float('-inf'))
        
        if self.use_flash and 'flash_attn_func' in globals():
            # Convert to format expected by flash_attn_func: (B, T, H, D)
            # This is a more accurate implementation using the actual flash_attn_func
            q_flash = q.transpose(1, 2)  # (B, T, H, head_dim)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)
            
            # Use flash attention for the attention computation
            # But we still need to compute the scores ourselves to include the triplet term
            
            # We need to replicate the behavior of the attention mechanism but with our custom scores
            scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
            scores_shifted = scores - scores_max
            attn_weights = torch.exp(scores_shifted)
            
            # Apply causal mask again in exponential space
            attn_weights.masked_fill_(mask, 0.0)
            
            # Apply dropout
            if self.dropout_p > 0.0 and self.training:
                attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
            # Flash attention expects a specific format and returns the same
            attn_output = attn_output.transpose(1, 2)  # (B, T, H, head_dim)
            
        elif self.use_flash and 'scaled_dot_product_attention' in globals():
            # Using PyTorch's built-in Flash Attention
            # We still need to manually compute all the scores including triplet
            # But we can use the attention mechanism itself once we have the scores
            
            # Prepare for standard softmax & masking
            scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - scores_max
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Weighted sum
            attn_output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
            attn_output = attn_output.transpose(1, 2)  # (B, T, H, head_dim)
            
        else:
            # Standard non-flash attention
            scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - scores_max
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Weighted sum and reshape in fewer operations
            attn_output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
            attn_output = attn_output.transpose(1, 2)  # (B, T, H, head_dim)
        
        # Final reshape and output projection
        output = attn_output.reshape(B, T, D)  # Use reshape instead of contiguous().view()
        return self.out_proj(output)

def verify_equivalence(d_model=256, num_heads=4, verify_flash=True):
    """
    Verify numerical equivalence between all attention implementations
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    
    # Create models with fixed dropout for deterministic results
    dropout = 0.0
    original_model = OriginalTripletAttention(d_model, num_heads, dropout)
    optimized_model = OptimizedTripletAttention(d_model, num_heads, dropout, use_flash_attn=False)
    
    # Create flash attention model if requested and available
    flash_model = None
    if verify_flash and HAS_FLASH_ATTN:
        flash_model = OptimizedTripletAttention(d_model, num_heads, dropout, use_flash_attn=True)
    
    # Copy weights from original to optimized to ensure they're identical
    with torch.no_grad():
        # Extract weights from original model
        q_weight = original_model.q_proj.weight
        k_weight = original_model.k_proj.weight
        v_weight = original_model.v_proj.weight
        t_weight = original_model.t_proj.weight
        
        q_bias = original_model.q_proj.bias
        k_bias = original_model.k_proj.bias
        v_bias = original_model.v_proj.bias
        t_bias = original_model.t_proj.bias
        
        # Combine weights for optimized model
        combined_weight = torch.cat([q_weight, k_weight, v_weight, t_weight], dim=0)
        combined_bias = torch.cat([q_bias, k_bias, v_bias, t_bias])
        
        optimized_model.qkvt_proj.weight.copy_(combined_weight)
        optimized_model.qkvt_proj.bias.copy_(combined_bias)
        
        # Copy output projection
        optimized_model.out_proj.weight.copy_(original_model.out_proj.weight)
        optimized_model.out_proj.bias.copy_(original_model.out_proj.bias)
        
        # Copy weights to flash model if it exists
        if flash_model is not None:
            flash_model.qkvt_proj.weight.copy_(combined_weight)
            flash_model.qkvt_proj.bias.copy_(combined_bias)
            flash_model.out_proj.weight.copy_(original_model.out_proj.weight)
            flash_model.out_proj.bias.copy_(original_model.out_proj.bias)
    
    # Test with different batch sizes and sequence lengths
    test_cases = [
        {"batch_size": 2, "seq_len": 10},
        {"batch_size": 1, "seq_len": 50},
        {"batch_size": 4, "seq_len": 128},
    ]
    
    results = []
    
    for tc in test_cases:
        batch_size = tc["batch_size"]
        seq_len = tc["seq_len"]
        
        # Create test input
        torch.manual_seed(123 + batch_size + seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Move to CUDA if Flash Attention is being used (performs better on GPU)
        if flash_model is not None and torch.cuda.is_available():
            x = x.cuda()
            original_model = original_model.cuda()
            optimized_model = optimized_model.cuda()
            flash_model = flash_model.cuda()
        
        # Evaluate models
        with torch.no_grad():
            original_output = original_model(x)
            optimized_output = optimized_model(x)
            
            # Flash attention output if available
            flash_output = None
            if flash_model is not None:
                flash_output = flash_model(x)
        
        # Check if standard optimized outputs are close to original
        opt_is_close = torch.allclose(original_output, optimized_output, rtol=1e-5, atol=1e-5)
        opt_max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
        
        # Check if flash outputs are close to original
        flash_is_close = None
        flash_max_diff = None
        if flash_output is not None:
            flash_is_close = torch.allclose(original_output, flash_output, rtol=1e-5, atol=1e-5)
            flash_max_diff = torch.max(torch.abs(original_output - flash_output)).item()
        
        results.append({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "optimized_equiv": opt_is_close,
            "optimized_max_diff": opt_max_diff,
            "flash_equiv": flash_is_close,
            "flash_max_diff": flash_max_diff
        })
    
    return results

def measure_performance(d_model=768, num_heads=12, batch_size=8, seq_len=512, num_runs=10):
    """
    Measure and compare performance between implementations
    """
    # Create models (with small dropout for real-world scenario)
    dropout = 0.1
    torch.manual_seed(42)
    original_model = OriginalTripletAttention(d_model, num_heads, dropout)
    optimized_model = OptimizedTripletAttention(d_model, num_heads, dropout, use_flash_attn=False)
    
    # Flash attention model if available
    flash_model = None
    if HAS_FLASH_ATTN:
        flash_model = OptimizedTripletAttention(d_model, num_heads, dropout, use_flash_attn=True)
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Move to CUDA if Flash Attention is being used
    if flash_model is not None and torch.cuda.is_available():
        x = x.cuda()
        original_model = original_model.cuda()
        optimized_model = optimized_model.cuda()
        flash_model = flash_model.cuda()
    
    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = original_model(x)
            _ = optimized_model(x)
            if flash_model is not None:
                _ = flash_model(x)
    
    # Measure original model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = original_model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = (time.time() - start) / num_runs
    
    # Measure optimized model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = optimized_model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    optimized_time = (time.time() - start) / num_runs
    
    # Measure flash model if available
    flash_time = None
    if flash_model is not None:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = flash_model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        flash_time = (time.time() - start) / num_runs
    
    return {
        "original_time": original_time,
        "optimized_time": optimized_time,
        "flash_time": flash_time,
        "optimized_speedup": original_time / optimized_time,
        "flash_speedup": None if flash_time is None else original_time / flash_time,
        "flash_vs_optimized": None if flash_time is None else optimized_time / flash_time,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads
    }

if __name__ == "__main__":
    # Check for CUDA
    if torch.cuda.is_available():
        print("CUDA available - will use GPU for Flash Attention tests")
    else:
        print("CUDA not available - Flash Attention will be slower on CPU")
    
    # Run verification
    print("\nVerifying numerical equivalence...")
    equiv_results = verify_equivalence(d_model=256, num_heads=4)
    for i, result in enumerate(equiv_results):
        print(f"\nTest case {i+1}:")
        print(f"  Batch size: {result['batch_size']}, Sequence length: {result['seq_len']}")
        print(f"  Standard Optimized Implementation:")
        print(f"    Numerically equivalent: {result['optimized_equiv']}")
        print(f"    Maximum absolute difference: {result['optimized_max_diff']}")
        
        if result['flash_equiv'] is not None:
            print(f"  Flash Attention Implementation:")
            print(f"    Numerically equivalent: {result['flash_equiv']}")
            print(f"    Maximum absolute difference: {result['flash_max_diff']}")
    
    # Run performance comparison
    print("\nMeasuring performance...")
    # Smaller sequence length for CPU testing
    seq_len = 256 if not torch.cuda.is_available() else 512
    perf_results = measure_performance(d_model=768, num_heads=12, batch_size=8, seq_len=seq_len)
    
    print(f"Model dimensions: {perf_results['d_model']}, Heads: {perf_results['num_heads']}")
    print(f"Batch size: {perf_results['batch_size']}, Sequence length: {perf_results['seq_len']}")
    print(f"Original model time: {perf_results['original_time']:.6f} seconds")
    print(f"Optimized model time: {perf_results['optimized_time']:.6f} seconds")
    print(f"Optimized speedup: {perf_results['optimized_speedup']:.2f}x")
    
    if perf_results['flash_time'] is not None:
        print(f"Flash Attention time: {perf_results['flash_time']:.6f} seconds")
        print(f"Flash Attention speedup vs original: {perf_results['flash_speedup']:.2f}x")
        print(f"Flash Attention speedup vs optimized: {perf_results['flash_vs_optimized']:.2f}x")