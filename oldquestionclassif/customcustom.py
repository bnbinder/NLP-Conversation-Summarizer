import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data
data = pd.read_csv('data.csv')
sentences = data['sentence'].values
labels = data['label'].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

maxlen = 100  # Adjust based on average sentence length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=maxlen),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')

# After training your model, add this function to make predictions

def predict_questions(model, tokenizer, questions, threshold=0.5):
    sequences = tokenizer.texts_to_sequences(questions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
    predictions = model.predict(padded_sequences)
    predicted_classes = (predictions > threshold).astype(int)
    return predicted_classes

# Example usage
if __name__ == "__main__":
    # Add some sample questions
    test_questions = ["Is this a question?", "This is a statement.", "Can you help me?", "Close the door."]
    
    # Get predictions
    results = predict_questions(model, tokenizer, test_questions)
    
    for question, prediction in zip(test_questions, results):
        print(f'Text: "{question}" - Predicted class: {prediction[0]}')

