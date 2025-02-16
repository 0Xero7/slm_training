import torch
import torch.nn as nn
import torch.nn.functional as F
from TripletAttention import TripletAttention  # Import your existing implementation
from transformers import AutoTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, dropout=0.3):
        super().__init__()
        self.attn = TripletAttention(d_model, dropout=dropout)       #    60 epoch = 154/247     // 20 epoch = 316.87, 272, 187.12
        # self.attn = cm.ClassicalMHA(d_model, dropout=dropout)        #    60 epoch = 164/268     // 20 epoch = 333.47, 299, 204.00
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection.
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward network with residual connection.
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, ff_hidden=512, max_seq_len=256, dropout=0.3):
        super().__init__()
        # Increase dropout significantly
        self.emb_dropout = nn.Dropout(0.3)  # New embedding dropout
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, ff_hidden, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)  # Add final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

        # Weight tying between embedding and output layer
        self.token_embed.weight = self.lm_head.weight
        
    def forward(self, input_ids):
        """
        input_ids: (B, T)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()
        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.emb_dropout(x)  # Apply dropout to embeddings
        # Add learned positional embeddings.
        x = x + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)  # Final layer norm
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        Autoregressively generate tokens given a prompt.
        input_ids: (B, T)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Ensure we don't exceed max_seq_len.
            x_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits = self.forward(x_cond)
            # Get logits for the last token.
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class ClozeEvaluator:
    def __init__(self, model_path="checkpoints/experimental_full_25epoch.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Initialize model
        self.model = LanguageModel(
            vocab_size=len(self.tokenizer),
            d_model=512,
            num_layers=12,
            ff_hidden=2048,
            max_seq_len=128,
            dropout=0.2
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        fixed_state_dict = {k.replace('_orig_mod.', ''): v 
                           for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(fixed_state_dict)
        self.model.eval()

    @torch.no_grad()
    def get_predictions(self, prefix, suffix="", top_k=5, temperature=0.7):
        """
        Get predictions for the next token given a prefix.
        
        Args:
            prefix (str): Text leading up to the prediction point
            suffix (str): Text after the prediction point (for evaluation)
            top_k (int): Number of top predictions to return
            temperature (float): Sampling temperature
            
        Returns:
            list: Top k predictions with their probabilities
        """
        # Tokenize prefix
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
        
        # Get model predictions
        logits = self.model(prefix_tokens)
        last_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_token_logits, dim=-1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode(idx)
            # Create complete text
            complete_text = prefix + token + suffix
            predictions.append({
                'token': token,
                'probability': prob.item(),
                'complete_text': complete_text
            })
            
        return predictions
    
    def evaluate_accuracy(self, test_pairs, temperature=0.7):
        """
        Evaluate model on pairs of (prefix, target word).
        
        Args:
            test_pairs (list): List of tuples (prefix, target_word, suffix)
            temperature (float): Sampling temperature
            
        Returns:
            dict: Evaluation metrics
        """
        correct_top1 = 0
        correct_top5 = 0
        mrr = 0.0
        
        for prefix, target, suffix in test_pairs:
            predictions = self.get_predictions(prefix, suffix, temperature=temperature)
            
            # Clean predictions and target for comparison
            clean_predictions = [p['token'].strip() for p in predictions]
            clean_target = target.strip()

            print(prefix, " -> ", clean_predictions[0])
            
            # Check top-1 accuracy
            if clean_predictions[0] == clean_target:
                correct_top1 += 1
                
            # Check top-5 accuracy
            if clean_target in clean_predictions:
                correct_top5 += 1
                    
            # Calculate MRR
            for i, pred in enumerate(clean_predictions, 1):
                if pred == clean_target:
                    mrr += 1.0 / i
                    break
        
        total = len(test_pairs)
        return {
            'top1_accuracy': correct_top1 / total,
            'top5_accuracy': correct_top5 / total,
            'mrr': mrr / total,
            'total_samples': total
        }

def main():
    # Example usage
    evaluator = ClozeEvaluator()
    
    # Single prediction example
    prefix = "The sky is"
    suffix = "mat."
    predictions = evaluator.get_predictions(prefix, suffix, temperature=0.7)
    print("\nPredictions for completion of:", prefix, "___", suffix)
    for pred in predictions:
        print(f"Token: {pred['token']}, Probability: {pred['probability']:.4f}")
        print(f"Complete text: {pred['complete_text']}")
    
    # Evaluation example
    test_pairs = [
        ("The cat sat on the ", "red", " mat."),
        ("I like to eat ", "eggs", " for breakfast."),
        ("The sky is ", "blue", " today.")
    ]
    metrics = evaluator.evaluate_accuracy(test_pairs)
    print("\nEvaluation Results:")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")
    print(f"Mean Reciprocal Rank: {metrics['mrr']:.4f}")

if __name__ == "__main__":
    main()