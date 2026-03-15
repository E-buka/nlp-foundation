from torch import nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.pad_idx=1
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=config["model"]["embedding_dim"],    
                                    padding_idx=self.pad_idx
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
        mask = (x != self.pad_idx).unsqueeze(-1).float()     # (batch, seq_len, 1)
        masked_embedding = embedding * mask
        summed = masked_embedding.sum(dim=1)                 # (batch, emb_dim)
        lengths = mask.sum(dim=1).clamp(min=1) 
        pooled_avg = summed / lengths
        logits = self.linear_relu(pooled_avg)
        return logits 
    