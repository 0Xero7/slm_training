# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import numpy as np
# import time
# from torch.cuda.amp import autocast, GradScaler
# from tqdm.auto import tqdm
# from datasets import load_dataset
# from transformers import AutoTokenizer

# # import EnchancedTextDataset
# from Blocks import LanguageModel
# import TripletAttention as ta
# import ClassicalMHA as cm

# # Load the WikiText-103-raw-v1 dataset.
# def LoadWikiText103(max_seq_len, batch_size):
#     dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
#     # For demonstration, we use a subset of the training and validation splits.
#     print(len(dataset["validation"]["text"]))
#     train_texts = dataset["train"]["text"][:50_000]  # adjust size as needed
#     valid_texts = dataset["validation"]["text"]

#     # Remove empty or very short lines.
#     train_texts = [t for t in train_texts if len(t) > 20]
#     valid_texts = [t for t in valid_texts if len(t) > 20]

#     # Use the GPT-2 tokenizer (adding a pad token if needed).
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#     vocab_size = len(tokenizer)

#     def tokenize_texts(texts):
#         tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_len)
#         return tokenized["input_ids"]

#     train_input_ids = tokenize_texts(train_texts)
#     valid_input_ids = tokenize_texts(valid_texts)

#     # Create PyTorch datasets.
#     class TextDatasetOld(torch.utils.data.Dataset):
#         def __init__(self, input_ids):
#             self.input_ids = input_ids
#         def __len__(self):
#             return len(self.input_ids)
#         def __getitem__(self, idx):
#             ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
#             return ids

#     train_dataset = TextDatasetOld(train_input_ids)
#     valid_dataset = TextDatasetOld(valid_input_ids)

#     def collate_fn(batch):
#         return torch.stack(batch)  # (B, T)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#     return tokenizer, vocab_size, train_dataset, valid_dataset, tokenize_texts, TextDatasetOld, collate_fn, train_loader, valid_loader

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def LoadWikiText103(max_seq_len, batch_size):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    # For demonstration, we use a subset of the training and validation splits.
    print(len(dataset["validation"]["text"]))
    
    # Use the GPT-2 tokenizer (adding a pad token if needed).
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)

    # Define tokenization function for the dataset.
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_len)

    # Filter out short texts.
    def filter_function(example):
        return len(example["text"]) > 20

    # Process training data.
    train_dataset = dataset["train"].select(range(50_000)).filter(filter_function).select(range(30_000))
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    train_dataset.set_format(type="torch", columns=["input_ids"])

    # Process validation data.
    valid_dataset = dataset["validation"].filter(filter_function)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True, num_proc=4)
    valid_dataset.set_format(type="torch", columns=["input_ids"])

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader
