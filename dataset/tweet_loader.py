from pandas import DataFrame
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

def load_csv_data(path_list:list[str], feature_col:str, label_col:str, **params) -> DataFrame: 
    ## load and concatenate data
    if not path_list:
        raise ValueError("Please provide data path as string or list of strings")
        
    elif isinstance(path_list, list):
        data = pd.DataFrame()
        while len(path_list) > 0: 
            try: 
                path = path_list.pop()
                data_ = pd.read_csv(path, **params) if params else pd.read_csv(path)
            except Exception as e: 
                print(f"Error loading {path}: {e}")
                data_ = None
            data = pd.concat([data, data_], axis=0, ignore_index=True)
        if data.shape == (0, 0):
            raise ValueError("Empty dataframe returned")
        return data[[feature_col, label_col]].sample(frac=1.0, ignore_index=True)
        
    elif isinstance(path_list, str):
        data = pd.read_csv(path_list, **params) if params else pd.read_csv(path_list)
        if data.shape == (0, 0):
            raise ValueError("Empty dataframe returned")
        return data[[feature_col, label_col]].sample(frac=1.0, ignore_index=True)
    else:
        raise ValueError("Please provide data path as string or list of strings")

def tokenize(text):
    return text.lower().split()

def build_vocab(dataset, max_size=None):
    counter = Counter()
    for text in dataset:
        counter.update(tokenize(text))

    vocab = {
    "<unk>": 0, # setting any word that is not in the vocabulary to zero
    "<pad>":1  # setting the padding index to 1
    }
    for word, _ in counter.most_common(max_size):
        if word not in vocab:
            vocab[word] = len(vocab)
        if len(vocab) == max_vocab_size: #max_vocab_size in config
            break

    return vocab


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch: 
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)

    padded_text = pad_sequence(text_list, batch_first=True, padding_value=1.0)
    return padded_text.to(device), torch.tensor(label_list, dtype=torch.float64).to(device)
  


def build_dataloaders(config, path_list, feature_col, label_col, **params):
    data = load_csv_data(path_list, feature_col, label_col, params) # can pass to config
    if not data:
        return None
    ## split data into train and validation data
    data.dropna(axis=0, inplace=True, ignore_index=True)
    data.drop_duplicates(inplace=True, ignore_index=True)
    
    train_data = data.sample(frac=0.7) # frac as arg
    val_data = data.drop(train_data.index)
    test_data = val_data.sample(frac=0.5) # frac as arg
    val_data = val_data.drop(test_data.index)
    
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    # build vocabulary from training set
    vocabulary = build_vocab(train_data[feature_col])
    
    # create dataset objects
    train_dataset = CustomTextData(train_data[feature_col], train_data[label_col], vocabulary)
    val_dataset = CustomTextData(val_data[feature_col], val_data[label_col], vocabulary)
    test_dataset = CustomTextData(test_data[feature_col], test_data[label_col], vocabulary)
    
    # create data loaders 
    train_loader = DataLoader(train_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
    val_loader =  DataLoader(val_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
    test_loader =  DataLoader(test_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
        
    return train_loader, val_loader, test_loader, vocabulary