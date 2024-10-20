"""
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

textt = """
#So, I was raised as a middle-class kid. And I am actually the only person on this stage who has a plan that is about lifting up the middle class and working people of America. I believe in the ambition, the aspirations, the dreams of the American people. And that is why I imagine and have actually a plan to build what I call an opportunity economy. Because here's the thing. We know that we have a shortage of homes and housing, and the cost of housing is too expensive for far too many people. We know that young families need support to raise their children. And I intend on extending a tax cut for those families of $6,000, which is the largest child tax credit that we have given in a long time. So that those young families can afford to buy a crib, buy a car seat, buy clothes for their children. My passion, one of them, is small businesses. I was actually -- my mother raised my sister and me but there was a woman who helped raise us. We call her our second mother. She was a small business owner. I love our small businesses. My plan is to give a $50,000 tax deduction to start-up small businesses, knowing they are part of the backbone of America's economy. My opponent, on the other hand, his plan is to do what he has done before, which is to provide a tax cut for billionaires and big corporations, which will result in $5 trillion to America's deficit. My opponent has a plan that I call the Trump sales tax, which would be a 20% tax on everyday goods that you rely on to get through the month. Economists have said that Trump's sales tax would actually result for middle-class families in about $4,000 more a year because of his policies and his ideas about what should be the backs of middle-class people paying for tax cuts for billionaires.
"""

def removeStopWords(text):
    filteredWords = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filteredWords)

print(removeStopWords(textt))
"""
import spacy
from collections import Counter
import numpy as np

nlp = spacy.load("en_core_web_sm")

def isSignificantSentence(sentence):
    doc = nlp(sentence)
    
    containsAction = any(token.pos_ == "VERB" for token in doc)
    containsSignificantNoun = any(token.pos_ == "NOUN" for token in doc)
    
    return containsAction or containsSignificantNoun

def extractKeyPoints(paragraph):
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
    
    topN = 5
    return [sentence for sentence, score in sortedSentences[:topN]]

paragraph = """
First of all, I have no sales tax. That's an incorrect statement. She knows that. We're doing tariffs on other countries. Other countries are going to finally, after 75 years, pay us back for all that we've done for the world. And the tariff will be substantial in some cases. I took in billions and billions of dollars, as you know, from China. In fact, they never took the tariff off because it was so much money, they can't. It would totally destroy everything that they've set out to do. They've taken in billions of dollars from China and other places. They've left the tariffs on. When I had it, I had tariffs and yet I had no inflation. Look, we've had a terrible economy because inflation has -- which is really known as a country buster. It breaks up countries. We have inflation like very few people have ever seen before. Probably the worst in our nation's history. We were at 21%. But that's being generous because many things are 50, 60, 70, and 80% higher than they were just a few years ago. This has been a disaster for people, for the middle class, but for every class. On top of that, we have millions of people pouring into our country from prisons and jails, from mental institutions and insane asylums. And they're coming in and they're taking jobs that are occupied right now by African Americans and Hispanics and also unions. Unions are going to be affected very soon. And you see what's happening. You see what's happening with towns throughout the United States. You look at Springfield, Ohio. You look at Aurora in Colorado. They are taking over the towns. They're taking over buildings. They're going in violently. These are the people that she and Biden let into our country. And they're destroying our country. They're dangerous. They're at the highest level of criminality. And we have to get them out. We have to get them out fast. I created one of the greatest economies in the history of our country. I'll do it again and even better.
"""
keyPoints = extractKeyPoints(paragraph)
print("Key Points Extracted:")
for point in keyPoints:
    print(f"- {point}")
