from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
import keras

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
maxlen = 500
batch_size = 64
embedding_dims = 200
filters = 100
kernel_size = 5
hidden_dims = 350
epochs = 100

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Add parts-of-speech to data (if desired, but we'll ignore this flag in this version)
pos_tags_flag = False  # Disabling for now as we load data from a CSV

# Function to load data from a CSV file
def load_encoded_data(data_split=0.8):
    # Path to your CSV file
    data_file = "data.csv"  # Update this to the actual file path
    
    # Load dataset from CSV (expecting "sentence" and "label" columns)
    data = pd.read_csv(data_file)
    
    # Split data into sentences and labels
    sentences = data['sentence'].values
    labels = data['label'].values  # 1 for question, 0 for non-question
    
    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=(1 - data_split), random_state=42)
    
    # Tokenize the text data
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words=5000)  # Adjust the vocab size if necessary
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

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings("glove.twitter.27B.200d.txt")  # Update with your GloVe file path

# Prepare embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dims))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# Ensure the labels are binary (not categorical)
print('y_train example:', y_train[:5])

if not load_model_flag:
    print('Constructing model!')

    # Build the CNN model
    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.Embedding(max_words, embedding_dims, input_length=maxlen))
    model.add(tensorflow.keras.layers.Dropout(0.2))
    
    model.add(tensorflow.keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(tensorflow.keras.layers.GlobalMaxPooling1D())
    
    model.add(tensorflow.keras.layers.Dense(hidden_dims))
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Activation('relu'))
    
    # Output layer for binary classification
    model.add(tensorflow.keras.layers.Dense(1))  # 1 output for binary (question or non-question)
    model.add(tensorflow.keras.layers.Activation('sigmoid'))  # Sigmoid for binary classification
    
    # Compile the model with binary cross-entropy loss
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
    model = tensorflow.keras.model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    
    # Compile the loaded model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

# Example: Test with custom sentences
def get_custom_test_comments():
    # Add some sample sentences
    test_sentences = ["Is this a question?", "This is a statement.", "Can you help me?", "Close the door."]
    test_labels = [1, 0, 1, 0]  # 1 for questions, 0 for non-questions
    
    return test_sentences, test_labels

# Test with custom sentences
test_comments, test_comments_category = get_custom_test_comments()

# Tokenize and encode the custom test sentences
tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words=5000)
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
