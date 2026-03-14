from collections import Counter
from torch.utils.data import DataLoader
import re 
from .make_data import collate_batch, CustomTextData, load_csv_data 


def tokenize(tweet):
    return re.findall(r"(?u)[#@]?\b\w\w+\b", tweet.lower())

def build_vocab(config, dataset):
    counter = Counter()
    for text in dataset:
        counter.update(tokenize(text))

    vocab = {
    "<unk>": 0, # setting any word that is not in the vocabulary to zero
    "<pad>":1  # setting the padding index to 1
    }
    for word, _ in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
        if len(vocab) == config["training"]["max_vocab_size"]: 
            break

    return vocab



def build_dataloaders(config, **params):
    data = None
    data = load_csv_data(config, **params) 
    
    if data is None:
        return None
    ## split data into train and validation data
    data.dropna(axis=0, inplace=True, ignore_index=True)
    data.drop_duplicates(inplace=True, ignore_index=True)
    
    train_data = data.sample(frac=config["data"]["train_frac"]) 
    val_data = data.drop(train_data.index)
    test_data = val_data.sample(frac=config["data"]["test_frac"]) 
    val_data = val_data.drop(test_data.index)
    
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    # build vocabulary from training set
    vocabulary = build_vocab(config, train_data[config["data"]["feature_col"]])
    
    # create dataset objects
    train_dataset = CustomTextData(train_data[config["data"]["feature_col"]], train_data[config["data"]["label_col"]], vocabulary, tokenize)
    val_dataset = CustomTextData(val_data[config["data"]["feature_col"]], val_data[config["data"]["label_col"]], vocabulary, tokenize)
    test_dataset = CustomTextData(test_data[config["data"]["feature_col"]], test_data[config["data"]["label_col"]], vocabulary, tokenize)
    
    # create data loaders 
    train_loader = DataLoader(train_dataset, collate_fn=collate_batch, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader =  DataLoader(val_dataset, collate_fn=collate_batch, batch_size=config["training"]["batch_size"])
    test_loader =  DataLoader(test_dataset, collate_fn=collate_batch, batch_size=config["training"]["batch_size"])
        
    return train_loader, val_loader, test_loader, vocabulary