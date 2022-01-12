![img](./img/main.jpg)

# Apple and Google products - Twitter Sentiment Analysis
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. 
It aims to analyze people's sentiments, attitudes, opinions emotions, etc. towards elements such as, products, individuals, topics ,organizations, and services

What is the problem? Why Twitter?
The problem is to understand your audience, keep on top of what’s being said about your brand – and your competitors – but also discover new trends in the industry. Twitter sentiment analysis allows you to keep track of what's being said about your product or service on social media, and can help you detect angry customers or negative mentions before they they escalate.


## Data source
Data source - tweet_product_company.csv
GloVe embeddings download link - https://nlp.stanford.edu/projects/glove/

![img](./img/plots.jpg)


## Data processing

- Combine labels "I can't tell" and 'No emotion toward brand or product' to one category "Neutral emotion"
- Break apart the data and the labels
- Categorize target labels {0: 'Negative emotion', 1: 'Neutral emotion', 2: 'Positive emotion'}
- Get class weights to address class imbalance {1: 0.6097668279806423, 2: 0.327540695116586, 0: 0.06269247690277167}
- Convert training data into tensors to feed into neural net with Tokenizer() and fit_on_texts()
- Transforms each text in texts to a sequence of integers with Tokenizer() and text_to_sequences()
- Truncate and pad input sequences to be all the same lenght vectors with pad_sequences()
- Train/test split
- Load GloVe file into dictionary
- Create word embedding/word context matrix
- Create embedding layer including embedding matrix for weights


## Notbook describition

#### Libraries used:

- nltk
- string

- sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
- sklearn.pipeline import Pipeline
- sklearn.metrics import accuracy_score

- sklearn.ensemble import RandomForestClassifier
- sklearn.naive_bayes import MultinomialNB
- sklearn.model_selection import train_test_split
- sklearn import preprocessing
- sklearn.utils.multiclass import unique_labels


- matplotlib.pyplot
- pandas
- numpy
- seaborn

- keras.utils import to_categorical
- keras.preprocessing.sequence import pad_sequences
- keras.preprocessing.text import Tokenizer
- keras import models
- from keras import layers
- from keras import optimizers
- from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D


#### Model architecture overview

- Model 1: Simple LSTM Model with regularization, increase dimensionality
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 24, 50)            507400    
    _________________________________________________________________
    lstm_13 (LSTM)               (None, 256)               314368    
    _________________________________________________________________
    dense_9 (Dense)              (None, 3)                 771       
    =================================================================
    Total params: 822,539
    Trainable params: 315,139
    Non-trainable params: 507,400
    _________________________________________________________________
    
    Training Accuracy: 0.8525
    Testing Accuracy:  0.6734

- Model 2: LSTM with regularization, reduce dimensionality

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 24, 50)            507400    
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 24, 50)            0         
    _________________________________________________________________
    lstm_16 (LSTM)               (None, 100)               60400     
    _________________________________________________________________
    dense_12 (Dense)             (None, 3)                 303       
    =================================================================
    Total params: 568,103
    Trainable params: 60,703
    Non-trainable params: 507,400
    _________________________________________________________________
    Training Accuracy: 0.7100
    Testing Accuracy:  0.6756

- Model 3: LSTM Layer Stacking

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 24, 50)            507400    
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 24, 50)            0         
    _________________________________________________________________
    lstm_17 (LSTM)               (None, 24, 80)            41920     
    _________________________________________________________________
    lstm_18 (LSTM)               (None, 20)                8080      
    _________________________________________________________________
    dense_13 (Dense)             (None, 3)                 63        
    =================================================================
    Total params: 557,463
    Trainable params: 50,063
    Non-trainable params: 507,400
    _________________________________________________________________
    
    Training Accuracy: 0.7210
    Testing Accuracy:  0.6905

- Model 4: GRU Layer Stacking

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 24, 50)            507400    
    _________________________________________________________________
    gru_11 (GRU)                 (None, 24, 64)            22080     
    _________________________________________________________________
    gru_12 (GRU)                 (None, 32)                9312      
    _________________________________________________________________
    dense_15 (Dense)             (None, 3)                 99        
    =================================================================
    Total params: 538,891
    Trainable params: 31,491
    Non-trainable params: 507,400
    _________________________________________________________________
    Training Accuracy: 0.7459
    Testing Accuracy:  0.6954

- Other models

    [('Random Forest', 0.6097668279806423),
     ('Support Vector Machine', 0.6097668279806423),
     ('Logistic Regression', 0.6097668279806423)]


## Sumarry

#### General Architecture

The inputs for each of the following models are our training data which consists of 7,273 with 20% withheld for validation. Each one of these observations contains 50 “features” which correspond to each word in the tweet. Any 0’s indicate the absence of a word.

Each model ends with a dense layer with 3 nodes, because we have 3 possible labels: positive, neutral, or negative. Because we one-hot encoded our labels, we use softmax for this multiclass classification problem to get a probability for each class. Additionally, we use accuracy as our metric, because this is a classification problem. When we use the predict method from Keras, we get a 3 element row vector for each input. Each element corresponds to a probability of one of the 3 labels. Therefore, the label with the highest probability is the predicted outcome. We compile each model with adam and categorical crossentropy.

#### Incorporating the GloVe

GloVe is defined to be an “unsupervised learning algorithm for obtaining vector representations for words”. Pre-trained word vectors data was downloaded from the Standford University website. The models specifically use the 50 -dimensional embeddings of 1.2M words from 2B tweets. This is represented in a txt file that was parsed to create an index that maps words to their vector representation. Using GloVe data improved the accuracy of the model about 3-4%.

#### Best model

All models perform quite well, however, Model 3 with two stacked LSTM layers, seems to have the best train/validation accuracy based on the training/validation results plot (history_3).

All models have some difficulties with predicting negative emotions due to the class imbalance - small amount of negative labels in the original dataset. In order to mitigate the class imbalace, we passed the pass the class_wieght argument to the fit() functions. This improved the results and reduced overfitting.

Other models performance was significatly lower than NN.

#### References

Generating Word Cloud in Python. GeeksforGeeks

https://www.geeksforgeeks.org/generating-word-cloud-python/

https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/

