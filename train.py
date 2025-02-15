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

import TripletAttention as ta
import ClassicalMHA as cm

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


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                 HYPER PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
max_seq_len = 128
d_model = 256
num_layers = 8
ff_hidden = 1024
lr = 1e-4  # Reduced learning rate
num_epochs = 20
batch_size = 128
grad_clip = 0.5  # Add gradient clipping
weight_decay = 0.01  # Reduced weight decay

torch.backends.cudnn.benchmark = True

# ---------------------------
# Data Preparation using WikiText-103
# ---------------------------
# Load the WikiText-103-raw-v1 dataset.
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# For demonstration, we use a subset of the training and validation splits.
print(len(dataset["validation"]["text"]))
train_texts = dataset["train"]["text"][:200_000]  # adjust size as needed
valid_texts = dataset["validation"]["text"][:5000]

# Remove empty or very short lines.
train_texts = [t for t in train_texts if len(t) > 20]
valid_texts = [t for t in valid_texts if len(t) > 20]

# Use the GPT-2 tokenizer (adding a pad token if needed).
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = len(tokenizer)

def tokenize_texts(texts):
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_len)
    return tokenized["input_ids"]

train_input_ids = tokenize_texts(train_texts)
valid_input_ids = tokenize_texts(valid_texts)

# Create PyTorch datasets.
class TextDatasetOld(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        return ids

train_dataset = TextDatasetOld(train_input_ids)
valid_dataset = TextDatasetOld(valid_input_ids)

def collate_fn(batch):
    return torch.stack(batch)  # (B, T)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class EnhancedTextDataset:
    def __init__(self, max_seq_len=128, batch_size=32, tokenizer_name="gpt2"):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # Load multiple datasets for diversity
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1")
        books = load_dataset("bookcorpus", split='train')
        
        # Prepare tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Process datasets
        self.train_data = self._prepare_data(
            wiki["train"]["text"] + books["text"],
            is_training=True
        )
        self.valid_data = self._prepare_data(
            wiki["validation"]["text"],
            is_training=False
        )
        
    def _clean_text(self, text):
        """Basic text cleaning"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove empty brackets
        text = re.sub(r'\[\]|\(\)|\{\}', '', text)
        return text.strip()
    
    def _is_valid_sample(self, text):
        """More sophisticated filtering"""
        if len(text) < 50:  # Increased minimum length
            return False
        if text.count(' ') < 10:  # Ensure enough words
            return False
        if len(text.split()) / len(text) < 0.15:  # Rough check for reasonable word length
            return False
        return True
    
    def _sliding_window(self, tokens, stride=None):
        """Create overlapping sequences for better context"""
        if stride is None:
            stride = self.max_seq_len // 2
            
        windows = []
        for i in range(0, max(1, len(tokens) - self.max_seq_len), stride):
            window = tokens[i:i + self.max_seq_len]
            if len(window) == self.max_seq_len:
                windows.append(window)
        return windows
    
    def _prepare_data(self, texts, is_training=True):
        """Process and filter data"""
        # Clean and filter texts
        filtered_texts = []
        for text in texts:
            if isinstance(text, str):
                text = self._clean_text(text)
                if self._is_valid_sample(text):
                    filtered_texts.append(text)
        
        # Tokenize with sliding windows for training
        all_token_sequences = []
        for text in filtered_texts:
            tokens = self.tokenizer.encode(text)
            if is_training:
                sequences = self._sliding_window(tokens)
            else:
                # For validation, just take the first max_seq_len tokens
                if len(tokens) >= self.max_seq_len:
                    sequences = [tokens[:self.max_seq_len]]
                else:
                    sequences = []
            all_token_sequences.extend(sequences)
        
        # Shuffle training data
        if is_training:
            random.shuffle(all_token_sequences)
        
        return all_token_sequences
    
    def get_dataloaders(self):
        """Create PyTorch DataLoaders"""
        class TextDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return torch.tensor(self.sequences[idx], dtype=torch.long)
        
        train_dataset = TextDataset(self.train_data)
        valid_dataset = TextDataset(self.valid_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, valid_loader


# ---------------------------
# Instantiate Model, Optimizer, and Loss
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    ff_hidden=ff_hidden,
    max_seq_len=max_seq_len,
    dropout=0.2  # Slightly reduced dropout
).to(device)

# Optimizer setup
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'layer_norm' in name or 'bias' in name or 'pos_embed' in name:  # Added pos_embed to no decay
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Add this before the training loop
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

torch.set_float32_matmul_precision('high')
model = torch.compile(model)


scaler = torch.amp.GradScaler('cuda')
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    start = time.time()
    
    # Training loop
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with autocast
        with autocast():
            logits = model(batch)
            logits = logits[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Unscale before clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer and scaler steps
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar
        current_loss = loss.item()
        total_loss += current_loss * batch.size(0)
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    # Step scheduler once per epoch
    scheduler.step()
    
    # Calculate training metrics
    avg_loss = total_loss / len(train_dataset)
    perplexity = math.exp(avg_loss)
    elapsed = time.time() - start
    
    # Validation loop
    model.eval()
    total_val_loss = 0.0
    val_pbar = tqdm(valid_loader, desc='Validation')
    
    with torch.no_grad():
        for batch in val_pbar:
            batch = batch.to(device)
            with autocast():
                logits = model(batch)
                logits = logits[:, :-1, :].contiguous()
                targets = batch[:, 1:].contiguous()
                val_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            total_val_loss += val_loss.item() * batch.size(0)
            val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})
    
    # Calculate validation metrics
    avg_val_loss = total_val_loss / len(valid_dataset)
    val_perplexity = math.exp(avg_val_loss)
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': best_val_loss,
        }, 'best_model.pt')
    else:
        patience_counter += 1
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_loss:.4f} - Train Perplexity: {perplexity:.2f}")
    print(f"Valid Loss: {avg_val_loss:.4f} - Valid Perplexity: {val_perplexity:.2f}")
    print(f"Time: {elapsed:.2f}s - LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping after {patience} epochs without improvement")
        break

# Load best model at the end
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nLoaded best model with validation loss: {checkpoint['loss']:.4f}")

def evaluate_on_wikitext103_test():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    test_texts = dataset["test"]["text"]
    
    test_texts = [t for t in test_texts if len(t) > 20]
    test_input_ids = tokenize_texts(test_texts)
    
    test_dataset = TextDatasetOld(test_input_ids)
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False, 
                           collate_fn=collate_fn)
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            batch = batch.to(device)
            with autocast():
                logits = model(batch)
                logits = logits[:, :-1, :].contiguous()
                targets = batch[:, 1:].contiguous()
                
                # Create mask to exclude padding tokens
                non_pad_mask = targets != 50256  # Exclude <|endoftext|> tokens
                
                # Only calculate loss on non-padding tokens
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                mask_flat = non_pad_mask.view(-1)
                
                # Get only the relevant logits and targets
                logits_filtered = logits_flat[mask_flat]
                targets_filtered = targets_flat[mask_flat]
                
                loss = criterion(logits_filtered, targets_filtered)
                
                # Count number of real tokens
                num_tokens = mask_flat.sum().item()
                
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    test_perplexity = math.exp(avg_loss)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")
    
    return test_perplexity

evaluate_on_wikitext103_test()

# ---------------------------
# Autoregressive Generation
# ---------------------------
model.eval()
prompt = "India is a"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\n--- Generated Text ---")
print(output_text)