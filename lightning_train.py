# import functools
# import time
# import math
# import optuna
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pytorch_lightning as pl
# from transformers import AutoTokenizer
# import numpy as np

# # Custom imports for your dataset/model modules.
# import WikiText103Loader as wiki
# from Blocks import LanguageModel, TransformerBlock

# # -----------------------------------------------------------
# #                      HYPER PARAMETERS
# # -----------------------------------------------------------
# max_seq_len = 128
# d_model = 492
# num_layers = 12
# ff_hidden = 2048
# lr = 1e-4  
# num_epochs = 20
# batch_size = 128
# grad_clip = 0.5  
# weight_decay = 0.01  
# use_experimental = True

# # -----------------------------------------------------------
# # Data Preparation using WikiText-103 (or WikiText-2 for budget)
# # -----------------------------------------------------------
# tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader = wiki.LoadWikiText103(max_seq_len, batch_size)

# # -----------------------------------------------------------
# # PyTorch Lightning Module with Updated Epoch Hooks
# # -----------------------------------------------------------
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

# # Make sure vocab_size is available in hyperparameters
# lit_model = LitLanguageModel(
#     vocab_size=vocab_size,
#     d_model=d_model,
#     num_layers=num_layers,
#     ff_hidden=ff_hidden,
#     max_seq_len=max_seq_len,
#     lr=lr,
#     weight_decay=weight_decay,
#     num_epochs=num_epochs,
#     use_experimental=use_experimental
# )
# lit_model.hparams.vocab_size = vocab_size

# # -----------------------------------------------------------
# # Trainer Setup with Callbacks
# # -----------------------------------------------------------
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     mode="min",
#     save_top_k=1,
#     filename="best_model"
# )
# early_stop_callback = EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     mode="min"
# )

# trainer = pl.Trainer(
#     max_epochs=num_epochs,
#     gradient_clip_val=grad_clip,  # Use automatic gradient clipping
#     callbacks=[checkpoint_callback, early_stop_callback],
#     precision="bf16-mixed",  # Options: "bf16", 16 (fp16), or 32 (default)
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     devices=1 if torch.cuda.is_available() else None,
#     log_every_n_steps=50,
# )

# from optuna.integration import PyTorchLightningPruningCallback
# from pytorch_lightning import Callback

# # class CustomPruningCallback(PyTorchLightningPruningCallback):
# #     def state_dict(self):
# #         # Return an empty state dict so Lightning doesn’t complain.
# #         return {}
# #     def load_state_dict(self, state_dict):
# #         # Do nothing on load.
# #         pass

# #     from pytorch_lightning import Callback

# class CustomPruningCallback(Callback):
#     def __init__(self, trial, monitor):
#         super().__init__()
#         self.trial = trial
#         self.monitor = monitor

#     def on_validation_end(self, trainer, pl_module):
#         # Retrieve the monitored metric.
#         current_score = trainer.callback_metrics.get(self.monitor)
#         if current_score is None:
#             return
#         # Optionally, check if the trial should be pruned.
#         if self.trial.should_prune():
#             raise optuna.TrialPruned()


# def objective(trial, use_experimental, d_model):
#     # Sample other hyperparameters.
#     lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
#     # d_model = trial.suggest_categorical("d_model", [256, 512, 768])
#     # num_layers = trial.suggest_int("num_layers", 4, 12, step=2)
#     # ff_hidden = trial.suggest_categorical("ff_hidden", [512, 1024, 2048])
    
#     # Instantiate your Lightning module with the sampled hyperparameters
#     model = LitLanguageModel(
#         vocab_size=vocab_size,
#         d_model=d_model,
#         num_layers=num_layers,
#         ff_hidden=ff_hidden,
#         max_seq_len=max_seq_len,
#         lr=lr,
#         weight_decay=weight_decay,
#         num_epochs=num_epochs,
#         use_experimental=use_experimental,
#     )
#     model.hparams.vocab_size = vocab_size  # ensure vocab_size is available
    
#     # Set up callbacks (you might want to adjust logger paths per sweep)
#     from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#     from optuna.integration import PyTorchLightningPruningCallback
    
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_loss",
#         mode="min",
#         save_top_k=1,
#         filename="best_model"
#     )
#     early_stop_callback = EarlyStopping(
#         monitor="val_loss",
#         patience=3,
#         mode="min"
#     )
#     pruning_callback = CustomPruningCallback(trial, monitor="val_loss") # PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         gradient_clip_val=grad_clip,
#         callbacks=[checkpoint_callback, early_stop_callback, pruning_callback],
#         precision="bf16",
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1 if torch.cuda.is_available() else None,
#         log_every_n_steps=50,
#         enable_progress_bar=False,  # disable progress bar if desired
#     )
    
#     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
#     val_loss = trainer.callback_metrics.get("val_loss")
#     return val_loss.item() if val_loss is not None else float("inf")

# # Create partial objective functions for each setting.
# objective_experimental = functools.partial(objective, use_experimental=True, d_model=492)
# objective_classical = functools.partial(objective, use_experimental=False)

# # Run the sweep for use_experimental=True.
# study_exp = optuna.create_study(direction="minimize")
# study_exp.optimize(objective_experimental, n_trials=1)
# print("Best trial for experimental model:")
# print(study_exp.best_trial)

# # Run the sweep for use_experimental=False.
# study_classical = optuna.create_study(direction="minimize")
# study_classical.optimize(objective_classical, n_trials=1)
# print("Best trial for classical model:")
# print(study_classical.best_trial)


# # # -----------------------------------------------------------
# # # Start Training
# # # -----------------------------------------------------------
# # trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


import functools
import time
import math
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
import numpy as np

# Custom imports for your dataset/model modules.
import WikiText103Loader as wiki
from Blocks import LanguageModel, TransformerBlock

# -----------------------------------------------------------
# Fixed Hyperparameters (Architecture remains constant aside from d_model adjustment)
# -----------------------------------------------------------
max_seq_len = 128
# For experimental: use d_model = 492; for classical: use d_model = 512 (adjust as needed)
CONSTANT_NUM_LAYERS = 12
CONSTANT_FF_HIDDEN = 2048
num_epochs = 20  # Total training epochs for each trial

# -----------------------------------------------------------
# Base Data Preparation (to get tokenizer and vocab size)
# -----------------------------------------------------------
# We call this once; individual trials will recreate the loaders with different batch sizes.
tokenizer, vocab_size, _, _, _, _ = wiki.LoadWikiText103(max_seq_len, batch_size=128)

# -----------------------------------------------------------
# Updated Lightning Module (accepting dropout as a parameter)
# -----------------------------------------------------------
class LitLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, num_layers, ff_hidden, max_seq_len, lr, weight_decay, num_epochs, use_experimental, dropout):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for logging and checkpointing
        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            ff_hidden=ff_hidden,
            max_seq_len=max_seq_len,
            dropout=dropout,             # Now configurable
            use_experimental=use_experimental
        )
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
        logits = logits[:, :-1, :].contiguous().view(-1, self.hparams.vocab_size)
        loss = self.criterion(logits, targets)
        perplexity = torch.exp(loss)
        # Log per-step metrics (aggregated automatically over the epoch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            train_loss_val = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
            train_epoch_perplexity = math.exp(train_loss_val)
            self.log("train_epoch_perplexity", train_epoch_perplexity, prog_bar=True)
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
        eta_min = self.hparams.lr * self.hparams.eta_min_ratio
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs, eta_min=eta_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# -----------------------------------------------------------
# Custom Pruning Callback (compatible with Lightning v2.0)
# -----------------------------------------------------------
from pytorch_lightning import Callback

class CustomPruningCallback(Callback):
    def __init__(self, trial, monitor):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# -----------------------------------------------------------
# Objective Function for Hyperparameter Sweep (Model Size Constant)
# -----------------------------------------------------------
def objective(trial, use_experimental):
    # Sample training dynamics hyperparameters.
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.3)
    grad_clip = trial.suggest_uniform("grad_clip", 0.1, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [128])
    eta_min_ratio = trial.suggest_uniform("eta_min_ratio", 0.05, 0.2)

    # Re-create data loaders with the sampled batch size.
    tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader = wiki.LoadWikiText103(max_seq_len, batch_size)

    # Set d_model based on the attention mechanism.
    # If using experimental, fix d_model to 492 to keep model size constant.
    if use_experimental:
        d_model = 492
    else:
        d_model = 512  # or another constant chosen for classical attention

    # Instantiate your Lightning module.
    model = LitLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=CONSTANT_NUM_LAYERS,
        ff_hidden=CONSTANT_FF_HIDDEN,
        max_seq_len=max_seq_len,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        use_experimental=use_experimental,
        dropout=dropout
    )
    model.hparams.vocab_size = vocab_size
    model.hparams.eta_min_ratio = eta_min_ratio

    # Set up callbacks.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    pruning_callback = CustomPruningCallback(trial, monitor="val_loss")
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gradient_clip_val=grad_clip,
        callbacks=[checkpoint_callback, early_stop_callback, pruning_callback],
        precision="bf16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss is not None else float("inf")

# -----------------------------------------------------------
# Create Partial Objectives for the Two Experimental Conditions
# -----------------------------------------------------------
objective_experimental = functools.partial(objective, use_experimental=True)
objective_classical = functools.partial(objective, use_experimental=False)

# -----------------------------------------------------------
# Run the Sweep for use_experimental=True.
# -----------------------------------------------------------
study_exp = optuna.create_study(direction="minimize")
study_exp.optimize(objective_experimental, n_trials=20)
print("Best trial for experimental model:")
print(study_exp.best_trial)

# -----------------------------------------------------------
# Run the Sweep for use_experimental=False.
# -----------------------------------------------------------
study_classical = optuna.create_study(direction="minimize")
study_classical.optimize(objective_classical, n_trials=20)
print("Best trial for classical model:")
print(study_classical.best_trial)
