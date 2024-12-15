from nltk.tokenize import sent_tokenize
from modelUtils import loadModelAndTokenizer, predictSentiment, extractKeyPoints, generateSummary, Llama3  # Import the function from the module
import string

modelBert, tokenizerBert, deviceBert = loadModelAndTokenizer()
files = open("wtfisshesayingbruh.txt", "w")
bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")

"""
while True:
    user_input  = input("input that stuff:    ")
    if user_input == "exit":
        break
    else:
        sentiment = predictSentiment(user_input , model, tokenizer, device)
        print(f"\n{user_input }\nPredicted sentiment: {sentiment}\n")
"""

moderator = ["DAVID MUIR", "LINSEY DAVIS"]
candidates = ["VICE PRESIDENT KAMALA HARRIS", "FORMER PRESIDENT DONALD TRUMP"]
endings = {'.', '?', '!'}
titles = {'Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Prof.'}

values = []
key = []
fileLen = 0

sentimentTrue = "ITS A QUESTION"
sentimentTrueCollect = []

collection = {}

tempKey = ""
tempVal = ""

def removePunc(text):
    trans = str.maketrans("", "", string.punctuation)
    return text.translate(trans)

def removeQuotes(text):
    return text.replace('"', '')#.replace("'", "")

def parseTextIntoSent(text):
    sentrevise = []
    sentences = sent_tokenize(text) 
    for i, sent in enumerate(sentences):
        sentt = removeQuotes(sent)
        sentt = removePunc(sentt)
        sentrevise.append(sentt)
    return sentrevise

with open("ABCdebateTranscript.txt", "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            fileLen += 1
            text = line.strip()
            key.append(text[:text.find(":")])
            values.append(text[text.find(":")+2:]) 

for i in range(0, fileLen):
    if key[i] in moderator:
        print("-----------" + key[i] + "-----------")
        array = parseTextIntoSent(values[i])
        for f in array:
            sentiment = predictSentiment(f, modelBert, tokenizerBert, deviceBert)
            print(f"\n{f}\nPredicted sentiment: {sentiment}\n")
            if sentiment == sentimentTrue and values[i] not in sentimentTrueCollect:
                tempKey = key[i]
                tempVal = values[i]
                print("hello   " + tempVal)
                sentimentTrueCollect.append(values[i])
                collection[values[i]] = {candidates[0]: [], candidates[1]: []}
    else:
        if len(tempVal) != 0:
            print("hello2   " + str(len(tempVal)))
            collection[tempVal][key[i]].append(values[i])
    #else:
    #    print(values[i])
        
    """
    if key[i] in moderator:
        sentiment = predictSentiment(values[i], model, tokenizer, device)
        print(f"\n{values[i]}\nPredicted sentiment: {sentiment}\n")    
    else:
        print(values[i])
    """

"""
for f in collection:
    temp = ""
    count = 0
    print(f"\n{f}\n")
    for i in collection[f][candidates[0]]:
        sentences = sent_tokenize(i) 
        for l in sentences:
            temp += l + " "
            count += 1
    print(candidates[0])
    for i in extractKeyPoints(temp, count):
        print("- " + i)
    temp = ""
    count = 0
    for i in collection[f][candidates[1]]:
        sentences = sent_tokenize(i) 
        for l in sentences:
            temp += l + " "
            count += 1
    print("\n" + candidates[1])
    for i in extractKeyPoints(temp, count):
        print("- " + i)
"""

for f in collection:
    temp = ""
    count = 0
    print(f"\n{f}\n")
    files.write(f"\n{f}\n")
    for i in collection[f][candidates[0]]:
        sentences = sent_tokenize(i) 
        for l in sentences:
            temp += l + " "
            count += 1
    summary = bot.getResponse("summarize: " + temp)
    files.write("orig : " + temp + "\n")
    files.write(candidates[0] + "\n")
    files.write(summary + "\n")
    print(candidates[0] + "\n")
    print(summary + "\n")
    
    temp = ""
    count = 0
    for i in collection[f][candidates[1]]:
        sentences = sent_tokenize(i) 
        for l in sentences:
            temp += l + " "
    summary = bot.getResponse("summarize: " + temp)
    files.write("orig : " + temp + "\n")
    files.write(candidates[1] + "\n")
    files.write(summary + "\n")
    print(candidates[1] + "\n")
    print(summary + "\n")
    
files.close()