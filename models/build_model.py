from .embedding_model import EmbeddingClassifier
from .lstm_model import LSTMClassifier

def build_model(model_name, vocab_size, config): #use yaml config
    if model_name == "embedding":
        return EmbeddingClassifier(vocabulary_size=vocab_size, 
                                   config=config
                                   )
    elif model_name == "lstm":
        return LSTMClassifier(vocabulary_size=vocab_size,
                              config = config
                              )
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}. \nPlease use embedding or lstm.")
