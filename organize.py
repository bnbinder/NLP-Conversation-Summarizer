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


moderator = ["DAVID MUIR", "LINSEY DAVIS"]
endings = {'.', '?', '!'}
values = []
key = []

def parseTextIntoSent(text):
    sentences = []
    minIndex = 0
    for index, char in enumerate(text):
        if char in endings:
            sentences.append(text[minIndex:index])
            minIndex = index + 2
    return sentences     

with open("ABCdebateTranscript.txt", "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            text = line.strip()
            key.append(text[:text.find(":")])
            values.append(text[text.find(":")+2:]) 


for i in range(10, 20):
    print("-----------" + key[i] + "-----------")
    if key[i] in moderator:
        array = parseTextIntoSent(values[i])
        for f in array:
            print(f)
    else:
        print(values[i])