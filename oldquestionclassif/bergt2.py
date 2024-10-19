import pandas as pd
import string
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text

bert_preprocess = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-12-h-768-a-12/2")

# Load your custom CSV
data = pd.read_csv("questions_and_contexts.csv")

# Preprocess the text column
def remove_punctuation(text):
    trans = str.maketrans("", "", string.punctuation)
    return text.translate(trans)

def remove_quotes(text):
    return text.replace('"', '').replace("'", "")

data["text"] = data["text"].map(remove_punctuation)
data["text"] = data["text"].map(remove_quotes)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['text'],data['isquestion'], test_size=0.2)

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

model.fit(X_train, y_train, epochs=10)


model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
cm

from matplotlib import pyplot as plt
import seaborn as sn

sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test, y_predicted))

reviews = [
    'This movie was a complete letdown. The plot was confusing, the characters were one-dimensional, and the climax felt forced. I expected much more from the hype surrounding it.',
    'When the film began, I was shocked to see it was filmed using a cheap video camera! In fact, the camera shakes and looks worse than the average home movie. Even direct to DVD films should have production values better than this!',
    'This film is a heartwarming tale that beautifully captures the essence of human emotions. The characters are relatable, and the storyline is both touching and inspiring. I left the theater with a smile on my face.',
    'This movie is nothing short of an incredible cinematic experience. The direction, cinematography, and music blend seamlessly to create a masterpiece. Every frame felt meticulously crafted, and the emotional depth of the characters stayed with me long after the credits rolled.',
]

review_pred = model.predict(reviews)
review_pred = review_pred.flatten()

review_pred = np.where(review_pred > 0.5, 'positive', 'negative')
review_pred