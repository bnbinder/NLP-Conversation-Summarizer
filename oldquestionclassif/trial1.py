from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

values = []
key = []

model = BertForSequenceClassification.from_pretrained('bert-question-classifier')
tokenizer = BertTokenizer.from_pretrained('bert-question-classifier')

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Question" if prediction == 1 else "Non-Question"

def parseTextIntoSent(text):
    sentences = []
    minIndex = 0
    for index, char in enumerate(text):
        if char == ".":
            sentences.append(text[minIndex:index])
            minIndex = index + 2
    return sentences        

with open("ABCdebateTranscript.txt", "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            text = line.strip()
            key.append(text[:text.find(":")])
            values.append(text[text.find(":")+2:]) 

"""
for i in range(0, len(values)):
    isQuestion = False
    if classify_text(values[i]) == "Question":
        isQuestion = True
    print(key[i] + "is question = " + str(isQuestion) + " = " + values[i])
    isQuestion = False
"""

for i in range(10, 20):
    array = parseTextIntoSent(values[i])
    for f in array:
        print(f)
        print(classify_text(f))