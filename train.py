import torch 
import json
import yaml

from .utils.seed import set_seed
from .dataset.tweet_loader import build_dataloaders
from .models.build_model import build_model
from .training.trainer import Trainer
from utils.metrics import evaluate_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main(config):

    set_seed(config["seed"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, vocab = build_dataloaders(config, path_list="")
    
    # saving the vocabulary dictionary
    with open("artifacts/vocab.json", "w") as f:
        json.dump(vocab, f)
    
    model = build_model(
        model_name = "lstm", 
        vocab_size = len(vocab),
        config=config,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']["lr"],
        weight_decay=config['training']["weight_decay"]
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    trainer = Trainer(model, optimizer, criterion, device)
    history = trainer.fit(train_loader, val_loader, epochs=config["training"]["epochs"])
    
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("Final test metrics:", test_metrics)