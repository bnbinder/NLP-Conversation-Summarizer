# Core Libraries
import torch
from torch.utils.data import DataLoader, Dataset

# Transformers
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Data handling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and Preprocess Data
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

data = {
    'text': [
        "What is your name",
        "He just bought a new car",
        "Can you pass me the salt",
        "They are having dinner at a fancy restaurant",
        "How old is your brother",
        "The cat is sleeping on the couch",
        "Do you know the way to the library",
        "She is preparing for her exam tomorrow",
        "Could you please explain this to me",
        "We are planning a trip to the mountains",
        "Where do you live",
        "He is watching a documentary about history",
        "Are you coming to the event next week",
        "She left her keys on the table",
        "Why is the sky blue",
        "They went for a walk in the evening",
        "When is the deadline for this project",
        "I just finished my workout",
        "What time is the game tonight",
        "He has been feeling tired lately",
        "Could you tell me more about the topic",
        "She is learning how to bake bread",
        "Do you want to go out for lunch",
        "He is reading a newspaper article",
        "How can I improve my presentation skills",
        "We are going to the grocery store",
        "Is there a way to fix this issue",
        "The new restaurant downtown is really popular",
        "What are your plans for the weekend",
        "She enjoys painting in her free time",
        "Can I borrow your pen for a moment",
        "He just finished writing his report",
        "Are we still meeting later today",
        "The sun is setting over the horizon",
        "Where should we meet for coffee",
        "She just started a new job",
        "Could you help me find my glasses",
        "They are hosting a barbecue this weekend",
        "Why did you choose this course",
        "He is working on his car in the garage",
        "What do you think about this movie",
        "The flowers in the garden are blooming beautifully",
        "Will you be joining us for dinner",
        "They have been traveling around the world",
        "How did you know about the event",
        "He is cleaning the house right now",
        "Is there anything I can do to help",
        "She loves playing the piano in her spare time",
        "When will you arrive at the airport",
        "They are waiting for their flight to board",
        "How can I access the online portal",
        "He is building a model airplane",
        "Could you bring me a glass of water",
        "The trees are losing their leaves as autumn approaches",
        "What are the benefits of this program",
        "She spent the whole day painting",
        "Can you tell me the time of the meeting",
        "He is playing soccer with his friends",
        "Why are they postponing the meeting again",
        "They are discussing the new company policy",
        "What would you like to do tomorrow",
        "She is preparing a special dinner for her guests",
        "Is there a reason for this delay",
        "The sky is clear and blue today",
        "How can I transfer money online",
        "He took a break from studying",
        "Why does the sun shine",
        "He left the office early today",
        "Can you meet me at the park tomorrow",
        "She is reading a new book",
        "Do you think we will finish on time",
        "It is a beautiful day outside",
        "What time does the train leave",
        "The project deadline was extended",
        "Should we order pizza for lunch",
        "They are moving to a new house next month",
        "How do I reset my password",
        "The meeting starts at noon",
        "Will the store be open later",
        "She is planning a vacation next week",
        "Do you have any recommendations for books",
        "I went grocery shopping yesterday",
        "Can you help me with my assignment",
        "The flowers in the garden are blooming",
        "What is your favorite movie",
        "He is studying for his exams",
        "Are you going to the concert this weekend",
        "She bought a new laptop last week",
        "Can we schedule the meeting for tomorrow",
        "He has been working on his presentation all day",
        "What do you think about the new policy",
        "The restaurant downtown serves great food",
        "Should I call you later",
        "She enjoys hiking in the mountains",
        "How do you feel about the changes",
        "He traveled to Europe last summer",
        "Can we meet at the coffee shop",
        "They are going to the beach next weekend",
        "Why are you leaving so early",
        "The children are playing in the park",
        "Do you agree with the decision",
        "She baked a cake for the party",
        "Is it possible to finish this project on time",
        "They adopted a puppy from the shelter",
        "How did you solve the problem",
        "I have been reading a new book recently",
        "Can you explain the reasons why we need to submit the project earlier than the initial deadline",
        "They decided to go on a road trip across the country to visit all the national parks and historical landmarks",
        "What are the benefits of using a renewable energy source compared to traditional fossil fuels in the long run",
        "He spent the entire afternoon organizing his bookshelves according to genre and author which took longer than expected",
        "How do you plan to improve the team's performance during the next quarter with the new strategies in place",
        "She enjoys spending her weekends hiking in the mountains and exploring new trails with her friends and family",
        "Why do you believe that artificial intelligence will have a significant impact on the job market in the coming decade",
        "The company announced that they will be implementing a new policy to promote a healthier work environment starting next month",
        "Could you provide a detailed explanation of how the new software system integrates with the existing infrastructure",
        "He has been working on his research paper for weeks but is still struggling to find reliable sources for his argument",
        "What are the main challenges that the organization is facing in terms of employee retention and workplace satisfaction",
        "She has been volunteering at the animal shelter for the past few months and has helped find homes for several stray animals",
        "How can we ensure that the new sustainability initiatives will be successfully implemented across all departments in the company",
        "They are planning to renovate their home by adding an extra room and remodeling the kitchen to make it more functional",
        "Do you think it is possible to complete the construction project ahead of schedule without compromising on quality or safety",
        "He has been training for the marathon for the past six months running every morning and gradually increasing his distance",
        "What factors contributed to the rapid growth of the technology sector in recent years and how will it evolve in the future",
        "She is considering applying for graduate school in the field of environmental science to further her understanding of climate change",
        "Is there any way to speed up the approval process for the budget proposal without overlooking important details",
        "They decided to spend their vacation traveling through Europe visiting several countries and experiencing different cultures",
        "How does the human brain process information differently when engaged in creative tasks versus logical problem solving",
        "He is thinking about switching careers from engineering to teaching because he wants to have a more meaningful impact on society",
        "Could you give me an estimate of how much time it will take to complete the project considering the current progress",
        "They are working on developing a new product that aims to revolutionize the market by offering a more affordable and efficient solution",
        "Why is it important to involve all stakeholders in the decision making process when implementing a new company policy",
        "She has been working late every night this week in order to finish the final draft of her novel before the publisher's deadline",
        "What steps should be taken to improve communication between different teams to ensure that projects are completed on time",
        "They just signed a contract to purchase a new home and are looking forward to moving into the neighborhood next month",
        "How does the increasing reliance on technology in education affect students ability to develop critical thinking and problem solving skills",
        "He has been saving money for years in order to buy his dream car which he plans to customize with the latest features and technology",
        "Could you explain how the changes to the tax law will affect small business owners and what they can do to minimize the impact",
        "They are planning to host a large event next summer to celebrate the company's anniversary and invite clients from around the world",
        "Why do experts believe that climate change is one of the most pressing issues facing humanity and what steps can be taken to mitigate it",
        "She has been managing multiple projects at once while also overseeing a team of new employees who are still in training",
        "What are the long term implications of the recent changes in government policies regarding healthcare and social welfare programs",
        "They are considering expanding their business into international markets in order to reach a wider audience and increase profits",
        "How do you expect the rise of automation to influence the global economy and what measures can be taken to ensure a smooth transition",
        "He is planning to write a book about his experiences traveling around the world and the lessons he learned from different cultures"
    ],
    'label': [  1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   1, 0, 1, 0]

}


# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create dataset format
train_dataset = {'text': train_texts, 'label': train_labels}
val_dataset = {'text': val_texts, 'label': val_labels}

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Convert to torch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# Step 4: Fine-Tune the Model
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Train the model
trainer.train()

# Step 5: Evaluate and Use the Model
# Save the model
model.save_pretrained('bert-question-classifier')
tokenizer.save_pretrained('bert-question-classifier')

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-question-classifier')
tokenizer = BertTokenizer.from_pretrained('bert-question-classifier')

# Inference
def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Question" if prediction == 1 else "Non-Question"

# Test the classifier
test_sentence = "I am a duck"
print(classify_text(test_sentence))  # Output: "Non Question"
