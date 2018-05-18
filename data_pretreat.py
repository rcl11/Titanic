import pandas as pd
import re


files = ["train","test","train_1","train_2"]

CSV_COLUMN_NAMES = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
CSV_COLUMN_NAMES_DATA = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    
all_titles = []
all_decks = []


def family_scale(row):
    if row['Family_Size'] == 0:
        return 'Alone'
    if row['Family_Size'] > 0 and row['Family_Size'] < 4:
        return 'Small'
    else:
        return 'Large'


for file in files:
    if file=='test':
        train = pd.read_csv(file+".csv", names=CSV_COLUMN_NAMES_DATA, header=0)
    else:
        train = pd.read_csv(file+".csv", names=CSV_COLUMN_NAMES, header=0)
    
    #First create some new variables (feature engineering)
    #add peoples titles
    for index, row in train.iterrows():
       title = ""
       title = row['Name'].split(",")
       title = title[1].split(" ")
       all_titles.append(title[1])
       train.set_value(index,'Title',title[1])

    #Add the deck (Cabin letter) 
    for index, row in train.iterrows():
        deck = ""
        deck = row['Cabin']
        if str(deck)!='nan': 
            #some have more than 1 cabin, just take the first for lack of a better idea
            train["Deck"] = str(deck)[0]
            train.set_value(index,'Deck', str(deck)[0])
            all_decks.append(str(deck)[0])
        else :
            train.set_value(index,'Deck', '')
   
    train['Family_Size']=train['SibSp']+train['Parch']

    #Slightly better variable grouping the low stats of those with large families
    train['Family_Scale'] = train.apply(lambda train: family_scale(train),axis=1)

    #Treat some missing values 

    train['Age'].fillna(int(train['Age'].mean()), inplace=True)
    train['Pclass'].fillna(train['Pclass'].mode(), inplace=True)
    train['SibSp'].fillna(train['SibSp'].mode(), inplace=True)
    train['Parch'].fillna(train['Parch'].mode(), inplace=True)

    #Use title info to make sensible guess on age where possible
    male_title = train['Title'].isin(['Rev.', 'Jonkheer.', 'Sir.', 'Capt.', 'Col.', 'Mr.', 'Dr.', 'the', 'Master.', 'Major.', 'Don.'])
    female_title = train['Title'].isin(['Miss.', 'Mme.', 'Dona.', 'Mlle.', 'Mrs.', 'Ms.', 'Lady.'])

    for index, row in train.iterrows():
        if row['Sex'] == '':
            if male_title.iloc[index]:
                train.set_value(index,'Sex','male')
            elif female_title.iloc[index]:
                train.set_value(index,'Sex','female')
            else:    
                train.set_value(index,'Sex','male')

    train.to_csv(file+"_corrected.csv")

print list(set(all_titles))
print list(set(all_decks))

