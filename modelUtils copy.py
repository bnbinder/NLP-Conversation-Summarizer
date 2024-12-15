import os
import torch
import transformers
from torch import nn
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from collections import Counter
import numpy as np

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def isSignificantSentence(sentence):
    doc = nlp(sentence)
    
    containsAction = any(token.pos_ == "VERB" for token in doc)
    containsSignificantNoun = any(token.pos_ == "NOUN" for token in doc)
    
    return containsAction or containsSignificantNoun

def extractKeyPoints(paragraph, topN):
    doc = nlp(paragraph)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    significantSentences = [sent for sent in sentences if isSignificantSentence(sent)]
    
    termFrequencies = Counter()
    for sent in significantSentences:
        words = nlp(sent)
        for token in words:
            if token.is_alpha and not token.is_stop: 
                termFrequencies[token.lemma_] += 1
    
    sentenceScores = {}
    for sent in significantSentences:
        score = sum(termFrequencies[token.lemma_] for token in nlp(sent) if token.lemma_ in termFrequencies)
        sentenceScores[sent] = score
    
    sortedSentences = sorted(sentenceScores.items(), key=lambda x: x[1], reverse=True)
    
    return [sentence for sentence, score in sortedSentences[:topN]]

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def loadModelAndTokenizer(model_path='bert_classifier.pth', bert_model_name='bert-base-uncased', num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    
    model.load_state_dict(torch.load(model_path))
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    return model, tokenizer, device

def predictSentiment(text, model, tokenizer, device, maxLength=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=maxLength, padding='max_length', truncation=True)
    inputIds = encoding['input_ids'].to(device)
    attentionMask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputIds, attention_mask=attentionMask)
        _, preds = torch.max(outputs, dim=1)
        logits = outputs.cpu().numpy()
        print("Logits flattened:", logits.flatten())
        realone = True if logits.flatten()[0] > 1.75 else False
        print("is it a question based on logit 1:", realone)
        realtwo = True if logits.flatten()[1] > 1.75 else False
        #print("is it a question based on logit 2:", realtwo)
        #return "ITS A QUESTION" if preds.item() == 1 else "NOPE"
        return "ITS A QUESTION" if logits.flatten()[1] > 1.75 else "NOPE"

def generateSummary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

class Llama3:
    def __init__ (self, modelPath):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model = modelPath,
            model_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )    
        self.terminator = self.pipeline.tokenizer.eos_token_id   
                
    def getResponse (self, query, maxTokens = 4096, temp = 0.6, topP = 0.9):
        userPrompt = [{"role": "system", "content": ""}] + [{"role": "user", "content": query}]        
        prompt = self.pipeline.tokenizer.apply_chat_template(
            userPrompt, 
            tokenize = False, 
            add_generation_prompt = True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens = maxTokens,
            eos_token_id = self.terminator,
            do_sample = True,
            temperature = temp,
            top_p = topP
        )
        response = outputs[0]["generated_text"][len(prompt):]    
        return response