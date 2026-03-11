from dataclasses import dataclass 
import json
import re
import torch
from torch.nn.utils.rnn import pad_sequence 

@dataclass 
class PredictionResult: 
    label: int = None
    probability: float = None
    
    def json_result(self):
        result = {
            "prediction_label":self.label, 
            "predcition_probability": self.probability,
        }
        return json.dumps(result)
    
    
def get_tweet(tweet: str):
    return re.findall("(?u)\w\w+", tweet.lower())
  
def numericalize(tokens):
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)
        
    token_indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    token_indices = torch.tensor(token_indices).long()
    return token_indices.unsqueeze(0)

def predict(tweet: str) -> json :
    tensor = numericalize(get_tweet(tweet))
    inputs = pad_sequence(tensor, batch_first=True)
    
    model =  model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)
    
    pred  = (probs >= 0.5).long().cpu().numpy()
        
    results = PredictionResult()
    results.label = pred
    results.probability = probs
    return results.json_result() 

    
    