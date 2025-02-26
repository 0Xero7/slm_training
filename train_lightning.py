import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoTokenizer
import numpy as np

# Custom imports for your dataset/model modules.
from datasets import load_dataset
import WikiText103Loader as wiki
from Blocks import LanguageModel, TransformerBlock

# -----------------------------------------------------------
#                      HYPER PARAMETERS
# -----------------------------------------------------------
max_seq_len = 256
d_model = 768
num_layers = 12
ff_hidden = 2048
lr = 1e-5
num_epochs = 1
batch_size = 64
grad_clip = 0.5  
weight_decay = 0.0009335091136195712
use_experimental = True
torch.set_float32_matmul_precision('high')
# -----------------------------------------------------------
# Data Preparation using WikiText-103 (or WikiText-2 for budget)
# -----------------------------------------------------------
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT")
# dataset.save_to_disk("/workspace/")


tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader = wiki.LoadFineWeb10B_Stream_OnTheFly(max_seq_len, batch_size, validation_split=0.01)
print(len(train_dataset), len(valid_dataset))

# -----------------------------------------------------------
# PyTorch Lightning Module with Updated Epoch Hooks
# -----------------------------------------------------------
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
        # Optionally compile the model if using PyTorch 2.0+
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
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
        
        # Log learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Optionally log gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, on_step=True, on_epoch=True, prog_bar=True)
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
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        # Calculate total steps
        total_steps = len(train_dataset) // batch_size * self.trainer.max_epochs
        warmup_steps = 2000
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.05,  # 5% warmup
            anneal_strategy='cos'
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# Make sure vocab_size is available in hyperparameters
lit_model = LitLanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    ff_hidden=ff_hidden,
    max_seq_len=max_seq_len,
    lr=lr,
    weight_decay=weight_decay,
    num_epochs=num_epochs,
    use_experimental=use_experimental
)
lit_model.hparams.vocab_size = vocab_size

# -----------------------------------------------------------
# Trainer Setup with Callbacks
# -----------------------------------------------------------
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project='112MFinewebEdu10B')

checkpoint_callback2 = ModelCheckpoint(
    monitor="val_loss",  # or another metric if you prefer
    mode="min",
    save_top_k=1,  # save only the best checkpoint
    filename="step-{step}-val_loss-{val_loss:.4f}"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    logger=wandb_logger,
    gradient_clip_val=grad_clip,  # Use automatic gradient clipping
    callbacks=[checkpoint_callback2, early_stop_callback],
    precision="bf16-mixed",  # Options: "bf16", 16 (fp16), or 32 (default)
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None,
    log_every_n_steps=50,
    val_check_interval=1000
)

# -----------------------------------------------------------
# Start Training
# -----------------------------------------------------------
trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            # ckpt_path='112MFinewebEdu10B/jcdgnaz8/checkpoints/step-step=12000-train_loss-train_loss=4.9217.ckpt')
