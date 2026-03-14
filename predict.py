from dataclasses import dataclass 
import json
import torch
from torch.nn.utils.rnn import pad_sequence 
from models.build_model import build_model
import yaml 
from pathlib import Path
from dataset.tweet_loader import tokenize

ROOT = Path(__file__).resolve().parent

with open(ROOT/"config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass 
class PredictionResult: 
    label: int = None
    probability: float = None
    
    def json_result(self):
        result = {
            "predicted_label":self.label, 
            "predicted_probability": self.probability,
        }
        return json.dumps(result)
 
with open(ROOT/"artifacts"/"vocab.json") as f:
    vocab = json.load(f)
    
model = build_model(
        model_name = "lstm", 
        vocab_size = len(vocab),
        config=config,
    )

model.load_state_dict(torch.load(ROOT/"models"/"best_model.pt", map_location=device))
model.to(device)
 
def numericalize(tokens, vocab):        
    token_indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    token_indices = torch.tensor(token_indices).long()
    return token_indices.unsqueeze(0)

def predict(tweet: str, tokenize, vocab, model) -> json :
    tensor = numericalize(tokenize(tweet), vocab)
    inputs = pad_sequence(tensor, batch_first=True)
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)
    
    pred  = (probs >= 0.5).long().cpu().item()
        
    results = PredictionResult()
    results.label = pred
    results.probability = round(probs.item(), 5)

    return results.json_result() 

    
if __name__ == "__main__":
    prediction = predict("come on, you are doing great",tokenize, vocab, model)
    print(prediction)