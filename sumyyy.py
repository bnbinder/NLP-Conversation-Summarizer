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

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# Download necessary resources (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def remove_fillers(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    filtered_sentences = []

    for sentence in sentences:
        # Tokenize each sentence into words
        words = word_tokenize(sentence)
        # Part-of-speech tagging
        tagged = pos_tag(words)
        
        # Filter out filler words based on their parts of speech
        # Keeping only nouns, verbs, adjectives, and adverbs
        filtered_words = [word for word, tag in tagged if tag.startswith(('N', 'V', 'J', 'R'))]

        # Reconstruct the sentence from filtered words
        if filtered_words:  # Only add non-empty sentences
            filtered_sentences.append(' '.join(filtered_words))

    # Join the filtered sentences into a final text
    return ' '.join(filtered_sentences)

# Sample paragraph
text = """
So, I was raised as a middle-class kid. And I am actually the only person on this stage who has a plan that is about lifting up the middle class and working people of America. I believe in the ambition, the aspirations, the dreams of the American people. And that is why I imagine and have actually a plan to build what I call an opportunity economy. Because here's the thing. We know that we have a shortage of homes and housing, and the cost of housing is too expensive for far too many people. We know that young families need support to raise their children. And I intend on extending a tax cut for those families of $6,000, which is the largest child tax credit that we have given in a long time. So that those young families can afford to buy a crib, buy a car seat, buy clothes for their children. My passion, one of them, is small businesses. I was actually -- my mother raised my sister and me but there was a woman who helped raise us. We call her our second mother. She was a small business owner. I love our small businesses. My plan is to give a $50,000 tax deduction to start-up small businesses, knowing they are part of the backbone of America's economy. My opponent, on the other hand, his plan is to do what he has done before, which is to provide a tax cut for billionaires and big corporations, which will result in $5 trillion to America's deficit. My opponent has a plan that I call the Trump sales tax, which would be a 20% tax on everyday goods that you rely on to get through the month. Economists have said that Trump's sales tax would actually result for middle-class families in about $4,000 more a year because of his policies and his ideas about what should be the backs of middle-class people paying for tax cuts for billionaires.
"""

# Remove fillers
cleaned_text = remove_fillers(text)
print(cleaned_text)


