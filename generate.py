# generate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from TripletAttention import TripletAttention  # Import your existing implementation

from Blocks import LanguageModel

def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    model = LanguageModel(
        vocab_size=len(tokenizer),
        d_model=400,
        num_layers=12,
        ff_hidden=2048,
        max_seq_len=128,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load("best_model.pt", map_location=device, weights_only=True)
    
    # Remove '_orig_mod.' prefix from compiled model keys
    fixed_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        fixed_state_dict[k.replace('_orig_mod.', '')] = v
    
    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model, tokenizer

def generate_text(prompt, max_length=50, temperature=1):
    model, tokenizer = load_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.lm_head.weight.device)
    generated = model.generate(inputs.input_ids, 
                             max_new_tokens=max_length,
                             temperature=temperature)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(generate_text("The sky is "))
