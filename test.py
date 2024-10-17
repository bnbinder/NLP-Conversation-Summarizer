import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as pit
import time
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from collections import Counter
from datasets import load_dataset
import re
import string

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
df = pd.read_csv("questions_and_contexts.csv")
# Display first few rows
print("head")
print(df.head())


# Stop words is a corpus of commonly used words such as the a an in that a search engine has
# been programmed to ignore, both when indexing entries for searching and when retrieving them
# as the result of a search query

# have a text column and a target column
"hello i am benjamin", 0
"fuck you?", 1
print("isquestion")
print((df.isquestion == 1).sum())  # boolean series sum of true which in this case is 1
print("isquestion")
print(df[df.isquestion == 1]) # boolean series of true values, prints those only

# string manipulation...

def removePunc(text):
    trans = str.maketrans("", "", string.punctuation)
    return text.translate(trans)

def removeStopWords(text):
    filteredWords = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filteredWords)

def removeQuotes(text):
    return text.replace('"', '').replace("'", "")


df["text"] = df.text.map(removePunc) #map(lambda x: functionForParsingString(x))
df["text"] = df.text.map(removeQuotes) #map(lambda x: functionForParsingString(x))

print("isquestion")
print((df.isquestion == 1).sum())  # boolean series sum of true which in this case is 1
print("text")
print(df[df.isquestion == 0]) # boolean series of true values, prints those only

# counting the words for tokenizer

def counterWords(textCol):
    count = Counter()
    for text in textCol.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counterWords(df.text)
fiveMostCommon = counter.most_common(5)
numUniqueWords = len(counter)

print("unique wordss " + str(numUniqueWords))

questions_df = df[df['isquestion'] == 1] 
answers_df = df[df['isquestion'] == 0]    

questions_train_size = int(questions_df.shape[0] * 0.8)
answers_train_size = int(answers_df.shape[0] * 0.8)

train_questions_df = questions_df[:questions_train_size]
val_questions_df = questions_df[questions_train_size:]

train_answers_df = answers_df[:answers_train_size]
val_answers_df = answers_df[answers_train_size:]

trainDf = pd.concat([train_questions_df, train_answers_df])
valDf = pd.concat([val_questions_df, val_answers_df])

trainSentences = trainDf.text.to_numpy()
trainLabels = trainDf.isquestion.to_numpy()

valSentences = valDf.text.to_numpy()
valLabels = valDf.isquestion.to_numpy()

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=numUniqueWords)
tokenizer.fit_on_texts(trainSentences) # fit only to training

# each word has a unique index
wordIndex = tokenizer.word_index

trainSequences = tokenizer.texts_to_sequences(trainSentences)
valSequences = tokenizer.texts_to_sequences(valSentences)

maxLength = 30 # max words in a sequence

trainPadded = tf.keras.preprocessing.sequence.pad_sequences(trainSequences, maxlen=maxLength, padding = "post", truncating = "post")
valPadded = tf.keras.preprocessing.sequence.pad_sequences(valSequences, maxlen=maxLength, padding = "post", truncating = "post")

reverseWordIndex = dict([(idx, word) for (word, idx) in wordIndex.items()])

def decode(sequence):
    return " ".join([reverseWordIndex.get(idx, "?") for idx in sequence])

#model LSTM
"""
model = keras.models.Sequential()
model.add(layers.Embedding(numUniqueWords,32,input_length=maxLength))
model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1,activation="sigmoid"))
model.build(input_shape=(None, maxLength))  # None for batch size, maxLength for sequence length
print("model summary")
print(model.summary())

loss = keras.losses.BinaryCrossentropy(from_logits = False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]
model.compile(loss=loss,optimizer=optim,metrics=metrics)
model.fit(trainPadded, trainLabels, epochs = 20, validation_data = (valPadded, valLabels), verbose = 2)
model_name      = "models/cnn_question_classifier"
"""

"""
loss = keras.losses.BinaryCrossentropy(from_logits = False)
model = keras.models.Sequential()
model.add(layers.Embedding(input_dim=numUniqueWords, output_dim=32, input_length=maxLength))
model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation='sigmoid'))
model.build(input_shape=(None, maxLength))  # None for batch size, maxLength for sequence length
model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
print("Model Summary:")
model.summary()
model.fit(trainPadded, trainLabels, epochs=20, validation_data=(valPadded, valLabels), verbose=2)
model_name      = "models/cnn_question_classifier3333"

try:
    model.save("my_model.h5")  
except:
    print("hiiii1111i")
try: 
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights(model_name + ".weights.h5")
    print("Saved model to disk")
except:
    print("hiiiii2")

predictions = model.predict(trainPadded)
predictions = [1 if p > 0.5 else 0 for p in predictions]


print("pred")
print(predictions[0:20])


"""



"""
model = load_model("my_model3333.h5")

predictions = model.predict(trainPadded)

print("pred")
print(predictions[0:20])

tokenizer = Tokenizer(num_words=numUniqueWords)  
tokenizer.fit_on_texts(trainSentences)  

def is_question(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    
    maxLength = 30  # Same as during training
    input_padded = pad_sequences(input_sequence, maxlen=maxLength, padding='post', truncating='post')

    prediction = model.predict(input_padded)

    return [0][0]  # 1 for question, 0 for context

while True:
    user_input = input("Enter text (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = is_question(user_input)
    print(result)


"""


print('Loading model!')
model_name      = "models/cnn_question_classifier3333"
maxlen = 500
batch_size = 64
embedding_dims = 75
filters = 100
kernel_size = 5
hidden_dims = 350
epochs = 1000
# Load model architecture and weights
json_file = open(model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights(model_name + ".weights.h5")
print("Loaded model from disk")

# Compile the loaded model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test set

# Example: Test with custom sentences
def get_custom_test_comments():
    # Add some sample sentences
    test_sentences = ["Is this a question", "This is a statement", "Can you help me", "Close the door"]
    test_labels = [1, 0, 1, 0]  # 1 for questions, 0 for non-questions
    
    return test_sentences, test_labels

# Test with custom sentences
test_comments, test_comments_category = get_custom_test_comments()

# Tokenize and encode the custom test sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(test_comments)
x_custom_test = tokenizer.texts_to_sequences(test_comments)
x_custom_test = tf.keras.preprocessing.sequence.pad_sequences(x_custom_test, maxlen=maxlen)


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
