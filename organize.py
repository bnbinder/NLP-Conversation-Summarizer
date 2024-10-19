"""

we have a text classifier, one and zero

only split sentences of moderators, which we need to define manually
keep text same for candidates

if moderator text is question
    summarize candidates responses
    break if encounter new question from moderator
    
new dictionary with one word description, with multi word description when found next question

economy - specific thing - gjireob jwen
        - speiciic thing - ngui howjubn 
        - specific thing - gnbuoejwr

"""
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from modelUtils import load_model_and_tokenizer, predict_sentiment  # Import the function from the module
import string

model, tokenizer, device = load_model_and_tokenizer()

moderator = ["DAVID MUIR", "LINSEY DAVIS"]
endings = {'.', '?', '!'}
titles = {'Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Prof.'}
values = []
key = []

def removePunc(text):
    trans = str.maketrans("", "", string.punctuation)
    return text.translate(trans)

def removeQuotes(text):
    return text.replace('"', '').replace("'", "")

def parseTextIntoSent(text):
    sentrevise = []
    sentences = nltk.sent_tokenize(text) 
    for i, sent in enumerate(sentences):
        sentt = removeQuotes(sent)
        sentt = removePunc(sentt)
        sentrevise.append(sentt)
    return sentrevise

with open("ABCdebateTranscript.txt", "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            text = line.strip()
            key.append(text[:text.find(":")])
            values.append(text[text.find(":")+2:]) 


for i in range(0, 50):
    print("-----------" + key[i] + "-----------")
    
    if key[i] in moderator:
        array = parseTextIntoSent(values[i])
        for f in array:
            sentiment = predict_sentiment(f, model, tokenizer, device)
            print(f"\n{f}\nPredicted sentiment: {sentiment}\n")
    #else:
    #    print(values[i])
    
    
    """
    if key[i] in moderator:
        sentiment = predict_sentiment(values[i], model, tokenizer, device)
        print(f"\n{values[i]}\nPredicted sentiment: {sentiment}\n")    
    else:
        print(values[i])
    """