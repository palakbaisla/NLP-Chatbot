import json #json module is used to work with json files in python
import pickle #pickle module is used save and load the module
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#loading the dataset 
with open('/Users/palakbaisla/Documents/Anzen_01/NLP/intents.json') as file:
    data=json.load(file)
   #the entire json file is loaded into data object  
#we are taking 2 empty lists they are the below ones:
patterns=[]
tags=[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])
        
#Train model
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(patterns)

model=LogisticRegression()
model.fit(X, tags)

#save model and vectorizer
pickle.dump(model,open("model.pkl",'wb'))
pickle.dump(vectorizer, open("vectorizer.pkl",'wb'))
print(" Model and Vectorizer saved sucessfully!")