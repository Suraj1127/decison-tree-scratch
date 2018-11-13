import pandas as pd

# Load the files
train = pd.read_csv('raw data/train.csv')
test = pd.read_csv('raw data/test.csv')

# Select appropriate columns for modelling.
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Map 1 to male and 0 to female
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})

# Save as csv file in preprocessed folder
train.to_csv('data/preprocessed/train.csv', index=False)