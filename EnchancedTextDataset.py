from datasets import load_dataset
from transformers import AutoTokenizer
import math

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