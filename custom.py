#nevermind it just upped accuracy but loss is still high, 85% acc 1.3 loss for first epoch, will let this run

from __future__ import print_function
from collections import Counter
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
import keras
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
numUniqueWords = 0
def counterWords(textCol):
    count = Counter()
    for text in textCol.values:
        for word in text.split():
            count[word] += 1
    return count

# Use can load a different model if desired
model_name      = "models/cnn_question_classifier"
load_model_flag = False
arguments       = sys.argv[1:len(sys.argv)]
if len(arguments) == 1:
    model_name = arguments[0]
    load_model_flag = os.path.isfile(model_name + ".json")
print(model_name)
print("Load Model?", (load_model_flag))

# Model configuration
maxlen = 300
batch_size = 128
embedding_dims = 42
filters = 100
kernel_size = 3
hidden_dims = 128
epochs = 3

# Add parts-of-speech to data (if desired, but we'll ignore this flag in this version)
pos_tags_flag = False  # Disabling for now as we load data from a CSV

# Function to load data from a CSV file
def load_encoded_data(data_split=0.8):
    
    global numUniqueWords
    
    
    
    
        
    stop = set(stopwords.words("english"))

    # tf.keras.preprocessing.text.Tokenizer
    # tf.keras.preprocessing.sequence.pad_sequences
    """
    raw_datasets = load_dataset("squad")

    # Extract questions and context from the dataset
    questions = raw_datasets["train"]["question"]
    contexts = raw_datasets["train"]["context"]  # Extract the context associated with each question

    # Create a list of tuples with questions (isquestion=1) and contexts (isquestion=0)
    data = []

    # Add questions with isquestion=1
    for question in questions:
        data.append((question, 1))

    # Add contexts with isquestion=0
    for context in contexts:
        data.append((context, 0))

    # Convert to a DataFrame
    df = pd.DataFrame(data, columns=['text', 'isquestion'])

    # Save the DataFrame to a CSV file
    df.to_csv("questions_and_contexts.csv", index=False)

    """
    data = pd.read_csv("questions_and_contexts.csv")
    #data = pd.read_csv("data.csv")
    # Display first few rows
    print("head")
    print(data.head())


    # Stop words is a corpus of commonly used words such as the a an in that a search engine has
    # been programmed to ignore, both when indexing entries for searching and when retrieving them
    # as the result of a search query

    # have a text column and a target column
    "hello i am benjamin", 0
    "fuck you?", 1
    print("isquestion")
    print((data.isquestion == 1).sum())  # boolean series sum of true which in this case is 1
    print("isquestion")
    print(data[data.isquestion == 1]) # boolean series of true values, prints those only
    

    counter = counterWords(data.text)
    fiveMostCommon = counter.most_common(5)
    numUniqueWords = len(counter)

    # string manipulation...

    def removePunc(text):
        trans = str.maketrans("", "", string.punctuation)
        return text.translate(trans)

    def removeStopWords(text):
        filteredWords = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(filteredWords)

    def removeQuotes(text):
        return text.replace('"', '').replace("'", "")

    def addQuotes(text):
        # Check if the text already starts and ends with quotes
        if not (text.startswith('"') and text.endswith('"')):
            return f'"{text}"'  # Add quotes if not present
        return text  # Return unchanged if already quoted

    data["text"] = data.text.map(removePunc) #map(lambda x: functionForParsingString(x))
    data["text"] = data.text.map(removeQuotes) #map(lambda x: functionForParsingString(x))

    data = data.sample(frac=1).reset_index(drop=True)

    print("isquestion")
    print((data.isquestion == 1).sum())  # boolean series sum of true which in this case is 1
    print("isquestion")
    print(data[data.isquestion == 1]) # boolean series of true values, prints those only
    
    
    # Split data into sentences and labels
    sentences = data['text'].values
    labels = data['isquestion'].values  # 1 for question, 0 for non-question
    
    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, train_size = data_split, test_size=(1 - data_split), random_state=42)
    #x_train, x_test, y_train, y_test = train_test_split(sentences, labels, train_size= 0.1, test_size=0.05, random_state=42)

        # Check the distribution of classes in the training set
    train_distribution = pd.Series(y_train).value_counts()
    print("Training Set Class Distribution:")
    print(train_distribution)

    # Check the distribution of classes in the test set
    test_distribution = pd.Series(y_test).value_counts()
    print("\nTesting Set Class Distribution:")
    print(test_distribution)
    
    # Tokenize the text data
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words=numUniqueWords)  # Adjust the vocab size if necessary
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    
    # Pad sequences to ensure uniform input size
    x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=maxlen)
    
    return x_train, x_test, y_train, y_test, tokenizer.word_index

# Load and prepare the data
x_train, x_test, y_train, y_test, word_index = load_encoded_data(data_split=0.8)

# Calculate the max number of words (for embedding layer input size)
max_words = len(word_index) + 1
num_classes = 1  # Binary classification (1 = question, 0 = non-question)

print(max_words, 'words')
print(num_classes, 'class')

# Ensure the labels are binary (not categorical)
print('y_train example:', y_train[:5])

if not load_model_flag:
    print('Constructing model!')

    # Build the CNN model
    model = tensorflow.keras.models.Sequential()

    #model.add(tensorflow.keras.layers.Embedding(numUniqueWords, embedding_dims, input_length=maxlen))
    #model.add(tensorflow.keras.layers.Dropout(0.4))
    
    #model.add(tensorflow.keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    #model.add(tensorflow.keras.layers.GlobalMaxPooling1D())
    
    #model.add(tensorflow.keras.layers.LSTM(hidden_dims, return_sequences=False))  # Set return_sequences=True if stacking LSTMs
    #model.add(tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(hidden_dims, return_sequences=False)))
    #model.add(tensorflow.keras.layers.CuDNNLSTM(hidden_dims, return_sequences=False))

    #model.add(tensorflow.keras.layers.Dense(hidden_dims, kernel_regularizer=tensorflow.keras.regularizers.l2(0.08)))
    #model.add(tensorflow.keras.layers.Dropout(0.4))
    #model.add(tensorflow.keras.layers.Activation('relu'))
    
    # Output layer for binary classification
    #  1 output for binary (question or non-question)
    #model.add(tensorflow.keras.layers.Dense(1,activation="sigmoid"))
    
    #second arch, less complicated
    """
    model.add(tensorflow.keras.layers.Embedding(numUniqueWords,embedding_dims,input_length=maxlen))
    #model.add(tensorflow.keras.layers.LSTM(hidden_dims, dropout=0.3, return_sequences=False))
    #model.add(tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(hidden_dims, dropout=0.3, return_sequences=True)))
    model.add(tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(hidden_dims, dropout=0.1, return_sequences=False)))
    model.add(tensorflow.keras.layers.Dense(1,activation="sigmoid")) 
    #model.add(tensorflow.keras.layers.Dense(hidden_dims, activation='relu'))
    #model.add(tensorflow.keras.layers.Dropout(0.4))
    #model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)))
    
    #model.build(input_shape=(None, maxlen))  # None for batch size, maxLength for sequence length
    # Compile the model with binary cross-entropy loss
    
    model.build(input_shape=(None, maxlen))  # None for batch size, maxLength for sequence length
    
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-4)  # Lower learning rate
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    """
    
    # arch type 3 from interwebs
    """
    #encoder = tensorflow.keras.layers.TextVectorization(
    #max_tokens=numUniqueWords)
    #encoder.adapt(x_train.map(lambda text, label: text))
    
    #model.add(encoder)
    
    model.add(tensorflow.keras.layers.Embedding(numUniqueWords,embedding_dims,input_length=maxlen))

    #model.add(tensorflow.keras.layers.LSTM(hidden_dims,activation='relu',return_sequences=True))

    #model.add(tensorflow.keras.layers.Dropout(0.2))

    model.add(tensorflow.keras.layers.LSTM(hidden_dims,return_sequences=False))

    #model.add(tensorflow.keras.layers.Dropout(0.2))

    # for units in [128,128,64,32]:

    # model.add(Dense(units,activation='relu'))

    # model.add(Dropout(0.2))

    model.add(tensorflow.keras.layers.Dense(hidden_dims,activation='relu'))

    #model.add(tensorflow.keras.layers.Dropout(0.2))

    model.add(tensorflow.keras.layers.Dense(1,activation='sigmoid'))
    
    model.build(input_shape=(None, maxlen))  # None for batch size, maxLength for sequence length

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    """
    
    #arch 4 i hate this
    """
    model.add(tensorflow.keras.layers.Embedding(numUniqueWords, embedding_dims))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    #model.add(tensorflow.keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    #model.add(tensorflow.keras.layers.GlobalAveragePooling1D())
    model.add(tensorflow.keras.layers.LSTM(hidden_dims,return_sequences=False))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    model.build(input_shape=(None, maxlen))  # None for batch size, maxLength for sequence length

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    """
    
    # arch 5, nope
    """
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Embedding(numUniqueWords, embedding_dims, input_length=maxlen))
    model.add(tensorflow.keras.layers.GRU(hidden_dims, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    #model.build(input_shape=(None, maxlen))  # None for batch size, maxLength for sequence length
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    """
    
    
    # A basic self-attention layer
    class SelfAttention(tensorflow.keras.layers.Layer):
        def __init__(self, units):
            super(SelfAttention, self).__init__()
            self.wq = tensorflow.keras.layers.Dense(units)
            self.wk = tensorflow.keras.layers.Dense(units)
            self.wv = tensorflow.keras.layers.Dense(units)

        def call(self, inputs):
            query = self.wq(inputs)
            key = self.wk(inputs)
            value = self.wv(inputs)

            attention_scores = tensorflow.matmul(query, key, transpose_b=True)
            attention_weights = tensorflow.nn.softmax(attention_scores, axis=-1)

            return tensorflow.matmul(attention_weights, value)

    # Usage in a model
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Embedding(numUniqueWords, embedding_dims, input_length=maxlen))
    model.add(SelfAttention(128))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Save model architecture and weights
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights(model_name + ".weights.h5")
    print("Saved model to disk")

else:
    print('Loading model!')

    # Load model architecture and weights
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tensorflow.keras.models.model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights(model_name + ".weights.h5")
    print("Loaded model from disk")
    
    # Compile the loaded model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

# Example: Test with custom sentences
def get_custom_test_comments():
    # Add some sample sentences
    test_sentences = ["Is this a question", "This is a statement", "Can you help me", "Close the door",
                      "Hello there general kenobi", "were you the chosen one", "do you know me"]
    test_labels = [1, 0, 1, 0, 0, 1, 1]  # 1 for questions, 0 for non-questions
    
    return test_sentences, test_labels

# Test with custom sentences
test_comments, test_comments_category = get_custom_test_comments()

# Tokenize and encode the custom test sentences
tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words=numUniqueWords)
tokenizer.fit_on_texts(test_comments)
x_custom_test = tokenizer.texts_to_sequences(test_comments)
x_custom_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_custom_test, maxlen=maxlen)

# Evaluate on the custom test set
predictions = model.predict(x_custom_test, batch_size=batch_size, verbose=1)

real = []
test = []
for i in range(0, len(predictions)):
    real.append(int(test_comments_category[i]))  # Use binary labels (not argmax)
    test.append(predictions[i])  # Sigmoid output > 0.5 is classified as 1 (question)

print("Predictions")
for i in range(len(predictions)):
    print(f"Text: \"{test_comments[i]}\" - Real: {real[i]}, Predicted: {test[i]}")