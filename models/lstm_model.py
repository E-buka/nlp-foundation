from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, 
                                      embedding_dim=config["model"]["embedding_dim"],
                                      padding_idx=1
                                      )
        
        self.lstm = nn.LSTM(input_size=config["model"]["embedding_dim"],
                        hidden_size=config["model"]["hidden_dim"],
                        num_layers=config["model"]["num_layers"],
                        dropout=config["model"]["dropout"],
                        batch_first=True
                        )
        self.fc = nn.Linear(in_features=config["model"]["hidden_dim"],
                        out_features=config["model"]["output_dim"])
    
    def forward(self, x):
        embedding = self.embedding(x)
        _, (hidden, _) = self.lstm(embedding)
        logits = self.fc(hidden[-1])
        return logits 
    
        