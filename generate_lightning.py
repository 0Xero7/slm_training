# import torch
# from transformers import AutoTokenizer
# import time
# import math
# import torch
# import pytorch_lightning as pl
# import torch.nn as nn
# import numpy as np
# from Blocks import LanguageModel, TransformerBlock
# import torch.optim as optim

# class LitLanguageModel(pl.LightningModule):
#     def __init__(self, vocab_size, d_model, num_layers, ff_hidden, max_seq_len, lr, weight_decay, num_epochs, use_experimental):
#         super().__init__()
#         self.save_hyperparameters()  # Saves hyperparameters for logging/checkpointing
#         self.model = LanguageModel(
#             vocab_size=vocab_size,
#             d_model=d_model,
#             num_layers=num_layers,
#             ff_hidden=ff_hidden,
#             max_seq_len=max_seq_len,
#             dropout=0.2,
#             use_experimental=use_experimental
#         )
#         # Optionally compile the model if using PyTorch 2.0+
#         # if hasattr(torch, "compile"):
#         #     self.model = torch.compile(self.model)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#         self.epoch_start_time = None

#     def forward(self, x):
#         return self.model(x)
    
#     def training_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         # Create targets by shifting tokens by one
#         targets = input_ids[:, 1:].contiguous().view(-1)
#         logits = self.model(input_ids)
#         # Align logits with targets by removing the last time-step
#         logits = logits[:, :-1, :].contiguous().view(-1, self.hparams.vocab_size)
#         loss = self.criterion(logits, targets)
#         perplexity = torch.exp(loss)
#         # Log per-step metrics (they are automatically aggregated over the epoch)
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True)
#         return loss

#     def on_train_epoch_start(self):
#         self.epoch_start_time = time.time()
    
#     def on_train_epoch_end(self):
#         # Retrieve the aggregated train loss from callback metrics
#         train_loss = self.trainer.callback_metrics.get("train_loss")
#         if train_loss is not None:
#             train_loss_val = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
#             train_epoch_perplexity = math.exp(train_loss_val)
#             self.log("train_epoch_perplexity", train_epoch_perplexity, prog_bar=True)
#         # Log the epoch time
#         if self.epoch_start_time is not None:
#             epoch_time = time.time() - self.epoch_start_time
#             self.log("epoch_time", epoch_time, prog_bar=True)
    
#     def validation_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         targets = input_ids[:, 1:].contiguous().view(-1)
#         logits = self.model(input_ids)
#         logits = logits[:, :-1, :].contiguous().view(-1, self.hparams.vocab_size)
#         loss = self.criterion(logits, targets)
#         perplexity = torch.exp(loss)
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_perplexity", perplexity, prog_bar=True)
#         return loss
    
#     def on_validation_epoch_end(self):
#         val_loss = self.trainer.callback_metrics.get("val_loss")
#         if val_loss is not None:
#             val_loss_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
#             val_epoch_perplexity = math.exp(val_loss_val)
#             self.log("val_epoch_perplexity", val_epoch_perplexity, prog_bar=True)
    
#     def configure_optimizers(self):
#         # Group parameters for weight decay as in your original setup.
#         decay_params = []
#         no_decay_params = []
#         for name, param in self.model.named_parameters():
#             if 'layer_norm' in name or 'bias' in name or 'pos_embed' in name:
#                 no_decay_params.append(param)
#             else:
#                 decay_params.append(param)
#         optim_groups = [
#             {'params': decay_params, 'weight_decay': weight_decay},
#             {'params': no_decay_params, 'weight_decay': 0.0}
#         ]
#         optimizer = optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}


# # Load the tokenizer (make sure to use the same tokenizer you used during training)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your custom tokenizer if applicable

# # Load your trained model checkpoint
# ckpt_path = "112MFinewebEdu10B/i0af8q66/checkpoints/step-step=66000-val_loss-val_loss=4.0876.ckpt"
# model = LitLanguageModel.load_from_checkpoint(ckpt_path, strict=True)
# model.eval()



# # If using GPU, move the model to CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define your prompt
# prompt = "The sky is"
# input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# # Set the number of tokens to generate
# max_generated_tokens = 50

# # Start with the prompt tokens
# generated_ids = input_ids.clone()

# # Autoregressive generation loop
# for _ in range(max_generated_tokens):
#     # Get model outputs (shape: [batch, sequence_length, vocab_size])
#     outputs = model(generated_ids)
#     # Extract logits for the last token in the sequence
#     next_token_logits = outputs[:, -1, :]
    
#     # Greedy decoding: choose the token with the highest probability
#     next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
#     # Alternatively, you can use sampling:
#     # probs = torch.softmax(next_token_logits, dim=-1)
#     # next_token_id = torch.multinomial(probs, num_samples=1)
    
#     # Append the predicted token to the sequence
#     generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
#     # Optional: stop if an end-of-sequence token is generated
#     if next_token_id.item() == tokenizer.eos_token_id:
#         break

# # Decode the generated token ids back into text
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("Generated Text:\n", generated_text)


import torch
from transformers import AutoTokenizer
import time
import math
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from Blocks import LanguageModel, TransformerBlock
import torch.optim as optim

class LitLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, num_layers, ff_hidden, max_seq_len, lr, weight_decay, num_epochs, use_experimental):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for logging/checkpointing
        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            ff_hidden=ff_hidden,
            max_seq_len=max_seq_len,
            dropout=0.2,
            use_experimental=use_experimental
        )
        # Do not compile during inference to avoid state dict key mismatches.
        # if hasattr(torch, "compile"):
        #     self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.epoch_start_time = None

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        # Create targets by shifting tokens by one
        targets = input_ids[:, 1:].contiguous().view(-1)
        logits = self.model(input_ids)
        # Align logits with targets by removing the last time-step
        logits = logits[:, :-1, :].contiguous().view(-1, self.hparams.vocab_size)
        loss = self.criterion(logits, targets)
        perplexity = torch.exp(loss)
        # Log per-step metrics (they are automatically aggregated over the epoch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        # Retrieve the aggregated train loss from callback metrics
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            train_loss_val = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
            train_epoch_perplexity = math.exp(train_loss_val)
            self.log("train_epoch_perplexity", train_epoch_perplexity, prog_bar=True)
        # Log the epoch time
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.log("epoch_time", epoch_time, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = input_ids[:, 1:].contiguous().view(-1)
        logits = self.model(input_ids)
        logits = logits[:, :-1, :].contiguous().view(-1, self.hparams.vocab_size)
        loss = self.criterion(logits, targets)
        perplexity = torch.exp(loss)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            val_epoch_perplexity = math.exp(val_loss_val)
            self.log("val_epoch_perplexity", val_epoch_perplexity, prog_bar=True)
    
    def configure_optimizers(self):
        # Group parameters for weight decay as in your original setup.
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if 'layer_norm' in name or 'bias' in name or 'pos_embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optim_groups, lr=self.hparams.lr, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs, eta_min=self.hparams.lr/10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# Load the tokenizer (make sure to use the same tokenizer you used during training)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your custom tokenizer if applicable
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def fix_compiled_checkpoint(state_dict):
    """Strip the extra wrapper keys from a compiled checkpoint.
    
    For example, if keys are like "model._orig_mod.layers.10.attn.t_proj.bias",
    we want to remove the "_orig_mod." part.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Replace "model._orig_mod." with "model." if it exists.
        new_key = key.replace("model._orig_mod.", "model.")
        new_state_dict[new_key] = value
    return new_state_dict

# Load the checkpoint
ckpt_path = "112MFinewebEdu10B/i0af8q66/checkpoints/step-step=66000-val_loss-val_loss=4.0876.ckpt"
# ckpt_path = "lightning_logs/version_12/checkpoints/step-step=30000-train_loss-train_loss=5.3039.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Fix the state dict keys
fixed_state_dict = fix_compiled_checkpoint(checkpoint["state_dict"])

# Define hyperparameters exactly as used during training.
hparams = {
    "vocab_size": len(tokenizer),
    "d_model": 768,
    "num_layers": 12,
    "ff_hidden": 2048,
    "max_seq_len": 1024,
    "lr": 3e-5,
    "weight_decay": 0.0009335091136195712,
    "num_epochs": 1,  # used during training
    "use_experimental": True
}

# Load your trained model checkpoint
# ckpt_path = "112MFinewebEdu10B/i0af8q66/checkpoints/step-step=66000-val_loss-val_loss=4.0876.ckpt"
# model = LitLanguageModel.load_from_checkpoint(ckpt_path, **hparams, strict=True)
# model.eval()

model = LitLanguageModel(**hparams)
model.load_state_dict(fixed_state_dict, strict=True)
model.eval()

# If using GPU, move the model to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your prompt
prompt = "Complete the sentence: The people are walking in the street with marching band. The men in green vest are walking on both sides of the parade. the man in neon green jacket"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Set the number of tokens to generate
max_generated_tokens = 100
temperature = 0.7  # lower temperature for less randomness; adjust as needed

generated_ids = input_ids.clone()

for _ in range(max_generated_tokens):
    outputs = model(generated_ids)
    next_token_logits = outputs[:, -1, :]
    
    # Apply temperature scaling
    scaled_logits = next_token_logits / temperature
    # Convert logits to probabilities
    probs = torch.softmax(scaled_logits, dim=-1)
    # Sample a token from the distribution
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
    if next_token_id.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)


# # Start with the prompt tokens
# generated_ids = input_ids.clone()

# # Autoregressive generation loop
# for _ in range(max_generated_tokens):
#     # Get model outputs (shape: [batch, sequence_length, vocab_size])
#     outputs = model(generated_ids)
#     # Extract logits for the last token in the sequence
#     next_token_logits = outputs[:, -1, :]
    
#     # Greedy decoding: choose the token with the highest probability
#     next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
#     # Alternatively, you can use sampling:
#     # probs = torch.softmax(next_token_logits, dim=-1)
#     # next_token_id = torch.multinomial(probs, num_samples=1)
    
#     # Append the predicted token to the sequence
#     generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
#     # Optional: stop if an end-of-sequence token is generated
#     if next_token_id.item() == tokenizer.eos_token_id:
#         break

# # Decode the generated token ids back into text
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("Generated Text:\n", generated_text)
