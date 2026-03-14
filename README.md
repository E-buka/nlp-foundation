## NLP-Foundation 
### Project summary
The objective of this project is to build a production style mini-NLP classifier. The dataset used for the project is a subset (ca 250k) of the tweet dataset from kaggle dataset.  The dataset however, is a cleaned subset of the data from the tweet-project repository. 
In the project, the tweet dataset was loaded, and split into three parts, 70% for training, 15% for validation and 15% for testing. A vocabulary set was built using the training dataset. A linear classifier with LSTM model was used to train the data for 20 epochs and training and validation results saved as json file in history directory. The trained model was loaded and used to predict the test data and results saved as json file in history directory.

#### Folder structure
 nlp-foundation/ 
    
data/
    tweet_pos_1.csv
    tweet_neg_2.csv

dataset/
    __ init__.py
    make_data.py
    tweet_loader.py

history/
    train_history.json
    test_metrics.json

models/ 
    __ init__.py
    build_model.py
    embedding_model.py
    lstm_model.py
    best_model.pt


training/
    __ init__.py
    trainer.py

utils/
    __ init__.py
    metrics.py
    seed.py

__ init__.py
train.py
predict.py
config.yaml

#### install
Install the dependencies by running requirements.txt file.

#### Train
All the model parameters are set in the configuration file - config.yaml. This includes the path to the data, model parameters and training epochs. The configuration parameters are imported automatically into the train.py file for model training.  
Two types of model has been defined for training and these are NN.embedding model and LSTM model.  Choose the model to train as "lstm" or "embedding" in the commented section of main() function within trian.py file. The script has been trained on lstm model.  The model with the best validation error is saved automatically in the models directory while the training and test metrics are saved in the history directory as json files.  

#### Predict
Run the predict.py file to predict sample texts. Run load_predictor() function to load the saved model, vocabulary and configurations.  . The model, vocabulary and configurations are passed to the predict() function with a sample tweet as string.  The prediction for the tweet is returned as a json string showing both the predicted label and probability. 

####  Prediction sample
{"predicted_label": 1, "predicted_probability": 0.71932}

#### Limitations
One of the limitations of this project is the use of cleaned dataset and custom tokenizer for training and tokenisation. The tokenizer used regular expression to capture some tweet unique semantics, however, the vocabulary was built from a cleaned dataset which did not take these semantics into account, such as: #, @, ', etc. These are common in everyday tweets and the model did not use tweet specific tokenizers.  The model was trained only for 20 epochs which is a small training cycle for a neural model.  The project does not include logging for ease of debugging, hyperparameter search or learning rate schedulign for optimised model training.  

#### Future Improvements
In the future, in order to capture tweet specific semantics, the raw dataset will be used in addition with tweet specific tokenizers will be used to build the vocabulary. The training epochs will be increased with the addition of early stopping, logging, learning rate scheduling and hyperparameter search. 