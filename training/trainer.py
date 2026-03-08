import torch
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer 
        self.device = device
        
    def training(self, dataloader):
        
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device) 
            
            self.optimizer.zero_grad()
            logits = self.model(inputs).reshape(-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >=0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += labels.size(0)
        
        return {
            "loss": total_loss/total_samples,
            "accuracy": total_correct/total_samples,
        }
        
    @torch.no_grad()
    def validate(self, dataloader):
        
        self.model.eval() 
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, labels in dataloader: 
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)
            
            logits = self.model(inputs).reshape(-1) 
            loss = self.criterion(logits, labels) 
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >=0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += labels.size(0)
                
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
            
            
    def fit(self, train_loader, val_loader, epochs):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        best_val_loss= float("inf")
        
        for epoch in range(epochs): 
            train_metrics = self.training(train_loader)
            val_metrics = self.validate(val_loader)
            
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss = {train_metrics['loss']:.4f} | "
                f"val_loss = {val_metrics['loss']:.4f} | "
                f"train_acc = {train_metrics['accuracy']:.4f} | "
                f"val_acc = {val_metrics['accuracy']:.4f}"
            )
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(self.model.state_dict(), "models/best_model.pt")
                
        return history 
