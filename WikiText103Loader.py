from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def LoadWikiText103(max_seq_len, batch_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
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
