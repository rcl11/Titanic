import pandas as pd
import re


files = ["train","test","train_1","train_2"]

CSV_COLUMN_NAMES = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
CSV_COLUMN_NAMES_DATA = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    
all_titles = []
all_decks = []

for file in files:
    if file=='test':
        train = pd.read_csv(file+".csv", names=CSV_COLUMN_NAMES_DATA, header=0)
    else:
        train = pd.read_csv(file+".csv", names=CSV_COLUMN_NAMES, header=0)
    
    #add peoples titles
    for index, row in train.iterrows():
       title = ""
       title = row['Name'].split(",")
       title = title[1].split(" ")
       all_titles.append(title[1])
       train['Title'] = title[1]

    #Add the deck (Cabin letter) 
    for index, row in train.iterrows():
        deck = ""
        deck = row['Cabin']
        if str(deck)!='nan': 
            #some have more than 1 cabin, just take the first for lack of a better idea
            train["Deck"] = str(deck)[0]
            all_decks.append(str(deck)[0])
        else :
            train["Deck"] = ''    
    
    train.to_csv(file+"_corrected.csv")

print list(set(all_titles))
print list(set(all_decks))





#train['Deck']=train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
