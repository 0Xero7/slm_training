from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def LoadWikiText103(max_seq_len, batch_size):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", streaming=True)

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
    train_dataset = dataset["train"] #.select(range(100_000)).filter(filter_function) #.select(range(30_000))
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


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.tensor(input_ids)
    return {"input_ids": input_ids}

def LoadFineWeb10B_Stream(max_seq_len, batch_size):
    # Load the fineweb 10B dataset in streaming mode.
    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", )
    dataset = load_from_disk("/workspace/fineweb-edu-10BT")
    
    
    # Load the GPT-2 tokenizer and add a pad token if it doesn't exist.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)
    
    # Define tokenization function.
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_seq_len)
    
    # Filter out texts that are too short.
    def filter_function(example):
        return len(example["text"]) > 20
    
    def batched_filter_function(batch):
        # Return a list of booleans, one per example in the batch.
        return [len(text) > 20 for text in batch["text"]]
    
    # Process the training data.
    train_dataset = dataset["train"].filter(batched_filter_function, batched=True, batch_size=1000, num_proc=6)
    # train_dataset = train_dataset.map(tokenize_function, batched=False)
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000, num_proc=6)

    
    # If a "validation" split is present, process it; otherwise, return an empty dataset.
    if "validation" in dataset:
        valid_dataset = dataset["validation"].filter(filter_function)
        valid_dataset = valid_dataset.map(tokenize_function, batched=False)
    else:
        print("No validation split found; returning an empty dataset for validation.")
        valid_dataset = Dataset.from_dict({"input_ids": []})
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    
    return tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk  # Ensure you have datasets installed

class OnTheFlyTokenizedDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Filter out examples with too short text.
        if len(example["text"]) <= 20:
            # Skip examples with very short text.
            return None
        tokenized = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len
        )
        tokenized["input_ids"] = torch.tensor(tokenized["input_ids"])
        return tokenized

def custom_collate_fn(batch):
    # Remove any None entries (i.e., skipped examples)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids}

def LoadFineWeb10B_Stream_OnTheFly(max_seq_len, batch_size, validation_split=0.1):
    # Load dataset from disk
    dataset = load_from_disk("/workspace/fineweb-edu-10BT")
    
    # Load tokenizer and add pad token if needed.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)
    
    # Get the training split.
    train_hf_dataset = dataset["train"]
    
    # Check if a validation split exists; if not, create one.
    if "validation" in dataset:
        valid_hf_dataset = dataset["validation"]
    else:
        print("No validation split found; performing a 90/10 train/validation split.")
        split_dataset = train_hf_dataset.train_test_split(test_size=validation_split, seed=42)
        train_hf_dataset = split_dataset["train"]
        valid_hf_dataset = split_dataset["test"]
    
    # Wrap datasets with on-the-fly tokenization.
    train_dataset = OnTheFlyTokenizedDataset(train_hf_dataset, tokenizer, max_seq_len)
    valid_dataset = OnTheFlyTokenizedDataset(valid_hf_dataset, tokenizer, max_seq_len)
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              collate_fn=custom_collate_fn, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                              collate_fn=custom_collate_fn, num_workers=4, shuffle=False)
    
    return tokenizer, vocab_size, train_dataset, valid_dataset, train_loader, valid_loader
