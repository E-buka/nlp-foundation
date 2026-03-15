import torch 
import json
import yaml

from utils.seed import set_seed
from dataset.tweet_loader import build_dataloaders
from models.build_model import build_model
from training.trainer import Trainer
from utils.metrics import evaluate_model
from pathlib import Path 

ROOT = Path(__file__).resolve().parent

(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "history").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

with open(ROOT/"config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main(config):

    set_seed(config["seed"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, vocab = build_dataloaders(config)
    
    # saving the vocabulary dictionary
    with open(ROOT/"artifacts"/"vocab.json", "w") as f:
        json.dump(vocab, f)
    
    model = build_model(
        model_name = config['selected_model'],
        vocab_size = len(vocab),
        config=config,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']["lr"],
        weight_decay=config['training']["weight_decay"]
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    trainer = Trainer(model, criterion, optimizer, device)
    history = trainer.fit(train_loader, val_loader, epochs=config["training"]["epochs"], ROOT=ROOT)
    
    model.load_state_dict(torch.load(ROOT/"models"/"best_model.pt", map_location=device))
    test_metrics = evaluate_model(model, test_loader, device)
    
    with open(ROOT/"history"/"train_history.json", "w") as h:
        json.dump(history, h)
        
    with open(ROOT/"history"/"test_metrics.json", "w") as t:
        json.dump(test_metrics, t)
        
    print("Final test metrics:", test_metrics)
    

if __name__ == "__main__":
    main(config)