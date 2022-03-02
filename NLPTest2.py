# Importing libraries
import os
import torch
import pandas as pd
from datetime import datetime
import spacy
import spacy_transformers

# Storing docs in binary format
from spacy.tokens import DocBin

#print(torch.cuda.is_available())
# Reading the dataset
df = pd.read_csv("all-data.csv", encoding='latin-1')
print(df.head())

print(df['Text'][2])

print(df['Text'][3])

print(df.shape)

#Splitting the dataset into train and test
train = df.sample(frac = 0.8, random_state = 25)
test = df.drop(train.index)

# Checking the shape

print(train.shape, test.shape)

import spacy
nlp=spacy.load("en_core_web_lg")

#Creating tuples
train['tuples'] = train.apply(lambda row: (row['Text'],row['Sentiment']), axis=1)
train = train['tuples'].tolist()
test['tuples'] = test.apply(lambda row: (row['Text'],row['Sentiment']), axis=1)
test = test['tuples'].tolist()
train[0]

# User function for converting the train and test dataset into spaCy document
def document(data):
#Creating empty list called "text"
  text = []
  for doc, label in nlp.pipe(data, as_tuples = True):
    if (label=='positive'):
      doc.cats['positive'] = 1
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 0
    elif (label=='negative'):
      doc.cats['positive'] = 0
      doc.cats['negative'] = 1
      doc.cats['neutral']  = 0
    else:
      doc.cats['positive'] = 0
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 1
#Adding the doc into the list 'text'
      text.append(doc)
  return(text)

# Calculate the time for converting into binary document for train dataset

start_time = datetime.now()

#passing the train dataset into function 'document'
train_docs = document(train)

#Creating binary document using DocBin function in spaCy
doc_bin = DocBin(docs = train_docs)

#Saving the binary document as train.spacy
doc_bin.to_disk("train.spacy")
end_time = datetime.now()

#Printing the time duration for train dataset
print('Duration: {}'.format(end_time - start_time))

# Calculate the time for converting into binary document for test dataset

start_time = datetime.now()

#passing the test dataset into function 'document'
test_docs = document(test)
doc_bin = DocBin(docs = test_docs)
doc_bin.to_disk("valid.spacy")
end_time = datetime.now()

#Printing the time duration for test dataset
print('Duration: {}'.format(end_time - start_time))

#Calculating the time for training the model
start_time = datetime.now()
# To train the model. Enabled GPU and storing the model output in folder called output_updated

os.system('python3 -m spacy train config.cfg --verbose --output ./output_updated')

end_time = datetime.now()
#Printing the time taken for training the model
print('Duration: {}'.format(end_time - start_time))

#text = "Australia’s largest airline temporarily lays off 2,500 employees"
# Loading the best model from output_updated folder
#nlp = spacy.load("./output_updated/model-best")
#demo = nlp(text)
#print(demo.cats)

#text1 = 'Apple earnings: Huge iPhone 12 sales beat analyst expectations'
#demo = nlp(text1) 
#print(demo.cats)