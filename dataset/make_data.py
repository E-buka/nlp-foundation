from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from pandas import DataFrame
import pandas as pd
import yaml 

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class CustomTextData(Dataset):
    def __init__(self, texts, labels, vocabulary):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        
    def __getitem__(self, index):
        tokens = self.texts[index].lower().split()
        text_indices = [self.vocabulary.get(tok, self.vocabulary["<unk>"]) for tok in tokens]
        return torch.tensor(text_indices), torch.tensor(self.labels[index])
    
    def __len__(self):
        return len(self.texts)
        
def collate_batch(batch):
    texts, labels = zip(*batch)
    padded_seq = pad_sequence(texts, batch_first=True, padding_val=1.0)
    return padded_seq, torch.tensor(labels)

def load_csv_data(config, **params) -> DataFrame: 
    ## load and concatenate data
    if not config["data"]["path"]:
        raise ValueError("Please provide data path as string or list of strings")
        
    elif isinstance(config["data"]["path"], list):
        data = pd.DataFrame()
        while len(config["data"]["path"]) > 0: 
            try: 
                path = config["data"]["path"].pop()
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

    