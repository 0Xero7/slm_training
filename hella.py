import torch
from transformers import AutoTokenizer
import time
import math
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from Blocks import LanguageModel, TransformerBlock
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

class LitLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, num_layers, ff_hidden, max_seq_len, lr, weight_decay, num_epochs, use_experimental):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for logging/checkpointing
        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            ff_hidden=ff_hidden,
            max_seq_len=max_seq_len,   # max_seq_len now set to 128 as per hparams
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
        # For multiple choice, batch["input_ids"] shape: (batch_size, num_choices, seq_length)
        input_ids = batch["input_ids"]  # shape: (B, num_choices, L)
        # We'll iterate over choices and compute scores for each.
        B, num_choices, L = input_ids.shape
        scores = []
        for i in range(num_choices):
            # Get logits for each candidate sequence
            logits = self.model(input_ids[:, i, :])
            # Use the last token's logit score as a proxy for candidate likelihood
            last_logits = logits[:, -1, :]  # shape: (B, vocab_size)
            score, _ = last_logits.max(dim=-1)  # shape: (B,)
            scores.append(score.unsqueeze(1))
        # Stack scores for each candidate: (B, num_choices)
        scores = torch.cat(scores, dim=1)
        preds = torch.argmax(scores, dim=1)
        acc = (preds == batch["label"]).float().mean()
        self.log("val_accuracy", acc, prog_bar=True)
        return {"val_accuracy": acc}
    
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
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Fix the state dict keys
fixed_state_dict = fix_compiled_checkpoint(checkpoint["state_dict"])

# Define hyperparameters exactly as used during training.
# Note: max_seq_len is now set to 128.
hparams = {
    "vocab_size": len(tokenizer),
    "d_model": 768,
    "num_layers": 12,
    "ff_hidden": 2048,
    "max_seq_len": 128,
    "lr": 3e-5,
    "weight_decay": 0.0009335091136195712,
    "num_epochs": 1,  # used during training
    "use_experimental": True
}

# Load your trained model checkpoint into your LightningModule
model = LitLanguageModel(**hparams)
model.load_state_dict(fixed_state_dict, strict=True)
model.eval()

# If using GPU, move the model to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


### Sampling Code with Sliding Window Generation ###
prompt = "Complete the sentence: The people are walking in the street with marching band. The men in green vest are walking on both sides of the parade. the man in neon green jacket"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Define generation parameters
max_generated_tokens = 10  # total tokens to generate
temperature = 0.7  # lower temperature for less randomness; adjust as needed
context_window = hparams["max_seq_len"]  # fixed attention window of 128

generated_ids = input_ids.clone()

for _ in range(max_generated_tokens):
    # For sliding window, take the last 'context_window' tokens as input.
    current_input = generated_ids[:, -context_window:]
    outputs = model(current_input)
    next_token_logits = outputs[:, -1, :]
    
    # Apply temperature scaling
    scaled_logits = next_token_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
    if next_token_id.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)


### Hellaswag Evaluation Code ###
class HellaSwagDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.data = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Hellaswag dataset has a context ("ctx"), candidate endings ("endings"), and a label ("label")
        context = item["ctx"]
        endings = item["endings"]
        # Convert label to int if it's a string
        label = int(item["label"])
        
        # Tokenize each candidate by concatenating context and candidate ending
        inputs = [self.tokenizer(context + " " + ending,
                                truncation=True,
                                max_length=self.max_length,
                                padding="max_length",
                                return_tensors="pt")
                for ending in endings]

        # Stack each candidate's input_ids and attention_mask to get shape: (num_choices, seq_length)
        input_ids = torch.stack([inp["input_ids"].squeeze(0) for inp in inputs])
        attention_mask = torch.stack([inp["attention_mask"].squeeze(0) for inp in inputs])

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": torch.tensor(label)}


    def __getitem__2(self, idx):
        item = self.data[idx]
        # Hellaswag dataset has a context ("ctx"), candidate endings ("endings"), and a label ("label")
        context = item["ctx"]
        endings = item["endings"]
        label = item["label"]
        
        # Tokenize each candidate by concatenating context and candidate ending
        inputs = [self.tokenizer(context + " " + ending,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding="max_length",
                                 return_tensors="pt")
                  for ending in endings]

        # Stack each candidate's input_ids and attention_mask to get shape: (num_choices, seq_length)
        input_ids = torch.stack([inp["input_ids"].squeeze(0) for inp in inputs])
        attention_mask = torch.stack([inp["attention_mask"].squeeze(0) for inp in inputs])

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": torch.tensor(label)}

# Load Hellaswag dataset validation split
hellaswag_dataset = load_dataset("hellaswag", split="validation")
val_dataset = HellaSwagDataset(hellaswag_dataset, tokenizer, max_length=128)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Evaluate the model on Hellaswag
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_dataloader:
        # For multiple-choice, input_ids shape: (B, num_choices, seq_length)
        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # shape: (B,)

        B, num_choices, seq_len = input_ids.shape
        scores = []
        for i in range(num_choices):
            # Pass each candidate separately through the model.
            logits = model(input_ids[:, i, :])
            # Use the last token's logit score as a simple proxy for candidate likelihood
            last_logits = logits[:, -1, :]  # shape: (B, vocab_size)
            score, _ = last_logits.max(dim=-1)  # shape: (B,)
            scores.append(score.unsqueeze(1))
        scores = torch.cat(scores, dim=1)  # shape: (B, num_choices)
        preds = torch.argmax(scores, dim=1)
        correct += (preds == labels).sum().item()
        total += B

hellaswag_accuracy = correct / total * 100
print(f"Hellaswag Accuracy: {hellaswag_accuracy:.2f}%")
