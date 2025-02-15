# generate.py
import torch
from transformers import AutoTokenizer

# Import model architecture components (must match training code)
import math
import torch.nn as nn

import TripletAttention as ta
import ClassicalMHA as cm



class TransformerBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, dropout=0.3):
        super().__init__()
        self.attn = ta.TripletAttention(d_model, dropout=dropout)       #    60 epoch = 154/247     // 20 epoch = 316.87, 272, 187.12
        # self.attn = cm.ClassicalMHA(d_model, dropout=dropout)        #    60 epoch = 164/268     // 20 epoch = 333.47, 299, 204.00
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model)
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
    def __init__(self, vocab_size, d_model=256, num_layers=4, ff_hidden=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        # Increase dropout significantly
        self.emb_dropout = nn.Dropout(0.3)  # New embedding dropout
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, ff_hidden, dropout=dropout) for _ in range(num_layers)
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


# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Initialize model with correct parameters
model = LanguageModel(
    vocab_size=len(tokenizer),
    d_model=128,
    num_layers=8,
    ff_hidden=256,
    max_seq_len=128,
    dropout=0.2
).to(device)

# Load saved weights
checkpoint = torch.load('checkpoints/experimental_200k_20epoch_checkpoint.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generation function
def generate_text(prompt, max_length=50, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "India is a"
    print("--- Generated Text ---")
    print(generate_text(prompt))