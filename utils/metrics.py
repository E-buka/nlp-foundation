from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch 

@torch.no_grad()
def evaluate_model(model, dataloader, device): 
    model.eval()
    
    all_labels = []
    all_preds = []
    for inputs, labels in dataloader: 
        inputs = inputs.to(device)
        logits = model(inputs).reshape(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy()
        
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        
    return{
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }