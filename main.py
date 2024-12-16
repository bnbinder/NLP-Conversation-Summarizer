from nltk.tokenize import sent_tokenize
from modelUtils import loadModelAndTokenizer, predictSentiment, Llama3  # Import the function from the module
import string

modelBert, tokenizerBert, deviceBert = loadModelAndTokenizer()
files = open("wtfisshesayingbruh.txt", "w")
bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")

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


# by ranking, which sucks
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

# first iteration

"""
  
Summarize all claims the speaker makes about anything. Categorize them with bullet points in categories. 
The response generated should only be this: the categories (EX: **Economy and Taxes**) and the bullet points
about everything they say that fits in those categories (EX: Give tax breaks to billionaires and big 
corporations, increasing the deficit by $5 trillion, Implement a 20%% sales tax on everyday goods, which 
would disproportionately affect middle-class families, etc). Everything said should be put in its own 
category. Do not start the response with "Here are the categorized claims:", only categories and bullet points
should be in the respons. Also, if you have a category that talks about the speakers claims and things the speaker
is saying about someone else, split the category into two categories, adding (self) with what the speaker is saying
about themselves and (opponent) with what the speaker is saying about the opponent. Keep all specific and relevant information
in the bullet points, dont leave anything out:

"""
# summary = bot.getResponse("Summarize all claims the speaker makes about anything. Categorize them with bullet points in categories. The response generated should only be this: the categories (EX: **Economy and Taxes**) and the bullet points about everything they say that fits in those categories (EX: Give tax breaks to billionaires and big corporations, increasing the deficit by $5 trillion, Implement a 20%% sales tax on everyday goods, which would disproportionately affect middle-class families, etc). Everything said should be put in its own category. Do not start the response with \"Here are the categorized claims:\", only categories and bullet points should be in the response. Also, if you have a category that talks about the speakers claims and things the speaker is saying about someone else, split the category into two categories, adding (self) with what the speaker is saying about themselves and (opponent) with what the speaker is saying about the opponent (EX: **Economy (self)** and **Economy (opponent)**). Keep all specific and relevant information in the bullet points, dont leave anything out: " + temp)


# second iteration
"""
Categorize the speaker's claims into topics using bullet points, grouping all relevant details under each category 
without leaving anything out. Split categories into (self) for claims about the speaker and (opponent) for 
claims about their opponent when applicable. Example: *Economy (self)* for category, and *sentence *sentence for 
bullet points. Omit any categories not mentioned. Keep all claims specific and detailed, and present them without 
additional commentary or introductions. Do not start the response with "Here are the categorized claims:", only categories and bullet points
should be in the response: 
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
    summary = bot.getResponse("Categorize the speaker's claims into topics using bullet points, grouping all relevant details under each category without leaving anything out. Split categories into “(self)” for claims about the speaker and “(opponent)” for claims about their opponent when applicable. Example: *Economy (self)* for category, and *sentence *sentence for bullet points. Omit any categories not mentioned. Keep all claims specific and detailed, and present them without additional commentary or introductions. Do not start the response with \"Here are the categorized claims:\", only categories and bullet points should be in the response:  " + temp)
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
    summary = bot.getResponse("Categorize the speaker's claims into topics using bullet points, grouping all relevant details under each category without leaving anything out. Split categories into “(self)” for claims about the speaker and “(opponent)” for claims about their opponent when applicable. Example: *Economy (self)* for category, and *sentence *sentence for bullet points. Omit any categories not mentioned. Keep all claims specific and detailed, and present them without additional commentary or introductions. Do not start the response with \"Here are the categorized claims:\", only categories and bullet points should be in the response:  " + temp)
    files.write("orig : " + temp + "\n")
    files.write(candidates[1] + "\n")
    files.write(summary + "\n")
    print(candidates[1] + "\n")
    print(summary + "\n")
    
files.close()