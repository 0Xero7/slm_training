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
import datetime

# import EnchancedTextDataset
import WikiText103Loader as wiki
from Blocks import LanguageModel
import TripletAttention as ta
import ClassicalMHA as cm
from stopwatch import Stopwatch

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                 HYPER PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
max_seq_len = 128
d_model = 512
num_layers = 12
ff_hidden = 2048
lr = 1e-4  # Reduced learning rate
num_epochs = 30
batch_size = 128
grad_clip = 0.5  # Add gradient clipping
weight_decay = 0.01  # Reduced weight decay
use_experimental = not True

torch.backends.cudnn.benchmark = True

# ---------------------------
# Data Preparation using WikiText-103
# ---------------------------
# tokenizer, vocab_size, train_dataset, valid_dataset, tokenize_texts, TextDatasetOld, collate_fn, train_loader, valid_loader = wiki.LoadWikiText103(max_seq_len, batch_size)
tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader = wiki.LoadWikiText103(max_seq_len, batch_size)

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
    dropout=0.2,  # Slightly reduced dropout
    use_experimental=use_experimental
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

log_file = open(f'logs/{datetime.datetime.now()}.log', 'a')
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
log_file.writelines([
    f'\nuse_experimental: {use_experimental}',
    f'\n--------------------------------------------------------------------------------',
    f'\nmax_seq_len: {max_seq_len}',
    f'\nd_model: {d_model}',
    f'\nnum_layers: {num_layers}',
    f'\nff_hidden: {ff_hidden}',
    f'\nlr: {lr}',
    f'\nnum_epochs: {num_epochs}',
    f'\nbatch_size: {batch_size}',
    f'\ngrad_clip: {grad_clip}',
    f'\nweight_decay: {weight_decay}',
    f'\n--------------------------------------------------------------------------------',
    f'\nModel Parameter Count: {params}'
])
log_file.flush()

# torch.set_float32_matmul_precision('high')
model = torch.compile(model)

scaler = torch.amp.GradScaler('cuda')
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    start = time.time()
    
    # ------------------------------------------------------------------------------------------------------------------------------
    #                                                       Training loop
    # ------------------------------------------------------------------------------------------------------------------------------

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    batch_load_timer = Stopwatch()
    transfer_timer = Stopwatch()
    zero_timer = Stopwatch()
    fwd_pass_timer = Stopwatch()
    backward_pass_timer = Stopwatch()
    optimizer_timer = Stopwatch()


    batch_load_timer.start()
    for batch in pbar:
        torch.cuda.synchronize() 
        batch_load_timer.stop()

        transfer_timer.start()    
        batch = batch["input_ids"].to(device, non_blocking=True)
        targets = batch[:, 1:].contiguous().view(-1)
        transfer_timer.stop()

        zero_timer.start()
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        torch.cuda.synchronize() 
        zero_timer.stop()
        
        # Forward pass with autocast
        fwd_pass_timer.start()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(batch)
            logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            loss = criterion(logits, targets)
        torch.cuda.synchronize() 
        fwd_pass_timer.stop()
        
        backward_pass_timer.start()
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        torch.cuda.synchronize() 
        backward_pass_timer.stop()

        # Unscale before clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer and scaler steps
        optimizer_timer.start()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize() 
        optimizer_timer.stop()
        
        # Update progress bar
        current_loss = loss.item()

        total_loss += current_loss * batch.size(0)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'batch_load': f'{batch_load_timer.elapsed():.4f}',
            'transfer': f'{transfer_timer.elapsed():.4f}',
            'zero': f'{zero_timer.elapsed():.4f}',
            'fwd_pass': f'{fwd_pass_timer.elapsed():.4f}',
            'bck_pass': f'{backward_pass_timer.elapsed():.4f}',
            'optimizer': f'{optimizer_timer.elapsed():.4f}',
        })

        batch_load_timer.start()
        
    
    # Step scheduler once per epoch
    scheduler.step()
    
    # Calculate training metrics
    avg_loss = total_loss / len(train_dataset)
    perplexity = math.exp(avg_loss)
    elapsed = time.time() - start
    
    log_file.writelines([
        f'\n--------------------------------------------------------------------------------',
        f'\nEpoch: {epoch + 1}/{num_epochs}',
        f'\nTime Taken: {elapsed:.2f}s - LR: {scheduler.get_last_lr()[0]:.2e}',
        f'\nTrain Loss: {avg_loss:.4f} - Train Perplexity: {perplexity:.2f}'
    ])
    log_file.flush()
    
    # ------------------------------------------------------------------------------------------------------------------------------
    #                                                       Validation loop
    # ------------------------------------------------------------------------------------------------------------------------------
    model.eval()
    total_val_loss = 0.0
    val_pbar = tqdm(valid_loader, desc='Validation')
    
    with torch.no_grad():
        for batch in val_pbar:
            batch = batch["input_ids"].to(device)
            with torch.amp.autocast('cuda'):
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
    log_file.writelines([
        f"\nValid Loss: {avg_val_loss:.4f} - Valid Perplexity: {val_perplexity:.2f}"  
    ])
    log_file.flush()

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"\nTotal batch load time: {batch_load_timer.elapsed()}")
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

# def evaluate_on_wikitext103_test():
#     dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
#     test_texts = dataset["test"]["text"]
    
#     test_texts = [t for t in test_texts if len(t) > 20]
#     test_input_ids = tokenize_texts(test_texts)
    
#     test_dataset = TextDatasetOld(test_input_ids)
#     test_loader = DataLoader(test_dataset, 
#                            batch_size=batch_size, 
#                            shuffle=False, 
#                            collate_fn=collate_fn)
    
#     model.eval()
#     total_loss = 0.0
#     total_tokens = 0
    
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Testing'):
#             batch = batch.to(device)
#             with autocast():
#                 logits = model(batch)
#                 logits = logits[:, :-1, :].contiguous()
#                 targets = batch[:, 1:].contiguous()
                
#                 # Create mask to exclude padding tokens
#                 non_pad_mask = targets != 50256  # Exclude <|endoftext|> tokens
                
#                 # Only calculate loss on non-padding tokens
#                 logits_flat = logits.view(-1, vocab_size)
#                 targets_flat = targets.view(-1)
#                 mask_flat = non_pad_mask.view(-1)
                
#                 # Get only the relevant logits and targets
#                 logits_filtered = logits_flat[mask_flat]
#                 targets_filtered = targets_flat[mask_flat]
                
#                 loss = criterion(logits_filtered, targets_filtered)
                
#                 # Count number of real tokens
#                 num_tokens = mask_flat.sum().item()
                
#             total_loss += loss.item() * num_tokens
#             total_tokens += num_tokens
    
#     avg_loss = total_loss / total_tokens
#     test_perplexity = math.exp(avg_loss)
    
#     print(f"\nTest Results:")
#     print(f"Test Loss: {avg_loss:.4f}")
#     print(f"Test Perplexity: {test_perplexity:.2f}")
    
#     return test_perplexity

# evaluate_on_wikitext103_test()

# # ---------------------------
# # Autoregressive Generation
# # ---------------------------
# model.eval()
# prompt = "India is a"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# generated = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
# output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
# print("\n--- Generated Text ---")
# print(output_text)