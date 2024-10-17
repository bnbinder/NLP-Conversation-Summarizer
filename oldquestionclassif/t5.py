from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Initialize T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Function to classify text using T5
def classify_text_t5(text):
    # Preprocess input text
    input_text = f"classify as question or non-question: {text}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate output (classification)
    outputs = model.generate(inputs['input_ids'], max_length=5)
    classification = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return classification

# Example data (replace with your dataset)
test_sentence = "I am a duck"
print(classify_text_t5(test_sentence))  # Output: "non-question"
