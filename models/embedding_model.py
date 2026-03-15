from torch import nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=config["model"]["embedding_dim"],    
                                    padding_idx=1
                                     )
        
        self.linear_relu = nn.Sequential(nn.Dropout(config["model"]["dropout"]),
                                         nn.Linear(config["model"]["embedding_dim"], 
                                                   config["model"]["hidden_dim"]),
                                         nn.ReLU(), 
                                         nn.Linear(config["model"]["hidden_dim"], 
                                                   config["model"]["output_dim"])
                                         )
        
    def forward(self, x):
        embedding = self.embedding(x)
        pooled_avg = embedding.mean(dim=1)
        logits = self.linear_relu(pooled_avg)
        return logits 
    