## NLP-Foundation 
### Objective
The objective of this project is to build a production style mini-NLP classifier.  The project builds upon my previous projects on ML and deployed pyspark model as found on weekly-challenge and tweet-project repositories. The dataset used for the project is a subset (ca 250k) of the tweet dataset from kaggle dataset.  However, the dataset used here is a cleaned dataset from the tweet-project repository which was subsampled for use ensuring a balance between positive and negative labels. Refer to week 7 of the weekly-challenge repository for a notebook style version of this NLP classification.  

#### Folder structure
 nlp-foundation/ 
    
data/

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
    best_model_emb.py

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

#### How to train
Once the model dependies are installed, all the model parameters are set in the configuration file - config.yaml. Here the path to the dataset, neural model parameters, training parameters and training epochs are set and can be modified. The configuration parameters are then imported into the train.py file. The train.py file is the training script.  
Two types of model has been defined for training and these are NN.embedding model and LSTM model.  In the train.py file, choose the model to train by typing "lstm" or "embedding" in the commented section of main() function definition. Then run the script. The model artifact with the best validation error is saved automatically in the models folder and the training and test metrics are saved to the history folder as json files. The train.py file can be run from CLI, vscode or any other support interface.  

#### How to predict
The predict.py file is the inference file for prediction.  The best trained model and vocabulary json file are loaded automatically in the script. The custom tokenization function is imported from the dataset folder. The model, vocabulary and tokenizer are passed to the predict() function with a sample tweet as string.  The prediction for the tweet is returned as a json string showing both the predicted label and probability. 
**sample output** 
{"predicted_label": 1, "predicted_probability": 0.71932}

#### Limitations
One obvious limitation of this project/model is the use of cleaned dataset and custom tokenizer for training and tokenisation. Although the tokenization function uses a regular expression to capture some tweet unique semantics, the vocabulary was built from a cleaned dataset which did not take these semantics into account, for example: #, @, ', etc. These are common in everyday tweets and the model did not use tweet specific tokenizers.  Also the model was trained only for 20 epochs because of dependency on a cpu device, and with limited dataset.  The project does not include logging for ease of debugging, and training times. However, to make up for these, the training metrics and test metrics are captured as json files in the history folder.  
Although these limitations exit, the objective of the project was still achieved for a mini-nlp classifier which has successfully trained and saved the best model artificat which can be deployed for prediction through an API. 