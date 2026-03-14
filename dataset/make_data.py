from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from pandas import DataFrame
import pandas as pd 



class CustomTextData(Dataset):
    def __init__(self, texts, labels, vocabulary, tokenize):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.tokenize = tokenize
        
    def __getitem__(self, index):
        tokens = self.texts.iloc[index]
        text_indices = [self.vocabulary.get(tok, self.vocabulary["<unk>"]) for tok in self.tokenize(tokens)]
        return torch.tensor(text_indices), torch.tensor(self.labels.iloc[index])
    
    def __len__(self):
        return len(self.texts)
        
def collate_batch(batch):
    texts, labels = zip(*batch)
    padded_seq = pad_sequence(texts, batch_first=True, padding_value=1)
    return padded_seq, torch.tensor(labels)

def load_csv_data(config, **params) -> DataFrame: 
    ## load and concatenate data
    if not config["data"]["path"]:
        raise ValueError("Please provide data path as string or list of strings")
        
    elif isinstance(config["data"]["path"], list):
        data = pd.DataFrame()
        for path in config["data"]["path"]: 
            try: 
                data_ = pd.read_csv(path, **params) if params else pd.read_csv(path)
            except Exception as e: 
                print(f"Error loading {path}: {e}")
                data_ = None
            data = pd.concat([data, data_], axis=0, ignore_index=True)
        if data.shape == (0, 0):
            raise ValueError("Empty dataframe returned")
        return data[[config["data"]["feature_col"], config["data"]["label_col"]]].sample(frac=1.0, ignore_index=True)
        
    elif isinstance(config["data"]["path"], str):
        data = pd.read_csv(config["data"]["path"], **params) if params else pd.read_csv(config["data"]["path"])
        if data.shape == (0, 0):
            raise ValueError("Empty dataframe returned")
        return data[[config["data"]["feature_col"], config["data"]["label_col"]]].sample(frac=1.0, ignore_index=True)
    else:
        raise ValueError("Please provide data path as string or list of strings")

    