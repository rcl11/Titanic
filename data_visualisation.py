import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

CSV_COLUMN_NAMES = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Title','Deck','Family_Size','Family_Scale']
NAN_VALUES = {'PassengerId':0, 'Survived':0,'Pclass':2,'Name':'','Sex':'male','Age':30,'SibSp':0,'Parch':0,'Ticket':'','Fare':10,'Cabin':'','Embarked':'','Title':'','Deck':'','Family_size':0}
       
train = pd.read_csv("train_corrected.csv", names=CSV_COLUMN_NAMES, header=0)
train.fillna(value=NAN_VALUES, inplace=True)

survived = train['Survived'] == 1
died = train['Survived'] == 0

survived_train = train[survived]
died_train = train[died]

ax1 = plt.subplot(211)
plt.ylabel("Passenger class")

male_survived = survived_train['Sex'] == 'male'
female_survived = survived_train['Sex'] == 'female'
male_died = died_train['Sex'] == 'male'
female_died = died_train['Sex'] == 'female'
male_survived_train = survived_train[male_survived]
male_died_train = died_train[male_died]
female_survived_train = survived_train[female_survived]
female_died_train = died_train[female_died]

plt.scatter(male_survived_train["Age"], male_survived_train["Pclass"], c='r', marker = 'o', label='Survived')
plt.scatter(male_died_train["Age"], male_died_train["Pclass"], c='b', marker = 'o', label='Died')
plt.title("male")
plt.axis([-5,110,0,4])
plt.legend()

ax2 = plt.subplot(212, sharex=ax1)

plt.xlabel("Age (years)")
plt.ylabel("Passenger class")
plt.scatter(female_survived_train["Age"], female_survived_train["Pclass"], c='r', marker = 'o', label='Survived')
plt.scatter(female_died_train["Age"], female_died_train["Pclass"], c='b', marker = 'o', label='Died')
plt.title("female")
plt.axis([-5,110,0,4])

plt.legend()
plt.savefig("AgePClassSex.png")


ax1 = plt.subplot(111)
plt.xlabel("Family Size")

x = np.array(survived_train["Family_Size"])
y = np.array(died_train["Family_Size"])
#elements, repeats = np.unique(x, return_counts=True)
plt.hist(x+0.3,bins=[0,0.2,1,1.2,2,2.2,3,3.2,4,4.2,5,5.2,6,6.2,7], histtype='bar', width=0.1, color='r', label='Survived')
plt.hist(y,bins=[0,0.2,1,1.2,2,2.2,3,3.2,4,4.2,5,5.2,6,6.2,7,7.2,8,8.2,9,9.2,10,10.2,11], histtype='bar', width=0.1, color='b', label='Died')
plt.legend()

plt.savefig("FamilySize.png")

