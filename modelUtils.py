import os
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

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

def load_model_and_tokenizer(model_path='bert_classifier.pth', bert_model_name='bert-base-uncased', num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_path))
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        logits = outputs.cpu().numpy()
        #print("Logits flattened:", logits.flatten())
        realone = True if logits.flatten()[0] > 1.5 else False
        #print("is it a question based on logit 1:", realone)
        realtwo = True if logits.flatten()[1] > 1.5 else False
        #print("is it a question based on logit 2:", realtwo)
        #return "ITS A QUESTION" if preds.item() == 1 else "NOPE"
        return "ITS A QUESTION" if logits.flatten()[1] > 1.5 else "NOPE"
