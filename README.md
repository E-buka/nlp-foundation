# NLP-Foundation: Tweet Sentiment Classifier

## Project Summary

This project builds a small production-style NLP text classifier for tweet sentiment prediction. It uses a cleaned subset of a Kaggle tweet dataset, prepared from the larger `tweet-project` repository.

The dataset is split into training, validation, and test sets in a 70/15/15 ratio. A vocabulary is built from the training set only. Two neural baseline architectures are implemented: an embedding-based classifier and an LSTM classifier. The selected model is trained for 20 epochs, the best model by validation loss is saved, and training history and test metrics are exported as JSON files.

## Folder Structure
```text
nlp-foundation/
├── artifacts/
│   └── vocab.json
├── data/
│   ├── tweet_neg_2.csv
│   └── tweet_pos_1.csv
├── dataset/
│   ├── __init__.py
│   ├── make_data.py
│   └── tweet_loader.py
├── history/
│   ├── test_metrics.json
│   └── train_history.json
├── models/
│   ├── __init__.py
│   ├── best_model.pt
│   ├── build_model.py
│   ├── embedding_model.py
│   └── lstm_model.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
|   ├── metrics.py
│   └── seed.py
├── config.yaml
├── predict.py
├── README.md
├── requirements.txt
└── train.py
```

## Installation

Create and activate a virtual environment, then install dependencies:

`bash`
pip install -r requirements.txt


## Training

All training parameters are defined in `config.yaml`, including data paths, model settings, and training hyperparameters.

Two model options are available:
- `embedding`
- `lstm`

Select the model to train inside `train.py` by setting the `model_name` argument in `build_model()`.

The training script:
- loads and splits the dataset
- builds the vocabulary from the training set
- trains the selected model
- saves the best model by validation loss to `models/best_model.pt`
- saves training history and final test metrics to the `history/` directory

Run training with:

`bash`
python train.py


## Prediction

The `predict.py` script loads the saved model, vocabulary, and configuration through `load_predictor()`. These are then used by the `predict()` function to generate predictions for input text.

The prediction output is returned as a JSON string containing:
- the predicted label
- the predicted probability

`bash`
python predict.py 


## Sample Prediction Output

{"predicted_label": 1, "predicted_probability": 0.71932}



## Limitations

This project uses a cleaned tweet dataset rather than raw tweet text. As a result, some tweet-specific signals such as hashtags, mentions, apostrophes, and other social-media-specific patterns are not fully represented in the training vocabulary.

Although a custom regex tokenizer is used, the preprocessing pipeline is still relatively simple and does not use a tweet-specific tokenizer designed for social media text.

In addition, the model was trained for only 20 epochs and does not yet include:
- early stopping
- learning rate scheduling
- hyperparameter tuning
- structured logging

## Future Improvements

Future versions of this project could be improved by:

- training on raw tweet text instead of heavily cleaned text
- using a tweet-specific tokenizer
- adding early stopping and learning rate scheduling
- introducing hyperparameter search
- improving experiment tracking and logging
- comparing stronger architectures and pretrained embeddings