import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("train.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

missing_percent = df.isnull().mean()*100
print(missing_percent)

df.drop(columns = ['Cabin'],inplace = True)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['PassengerId','Name','Ticket'],inplace=True)

df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first = True)

print(df.isnull().sum())
print(df.head())


print(df['Survived'].value_counts(normalize=True))

sns.barplot(x='Sex',y = 'Survived',data = df)
plt.title("Survival Rate by gender")
plt.show()

sns.barplot(x ='Pclass',y='Survived',data = df)
plt.title("Survival rate by class")
plt.show()

sns.histplot(df['Age'],kde=True)
plt.title("Age distribution")
plt.show()

numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot  = True,cmap='coolwarm')
plt.title("Correlation Heatmap(Numeric Features only)")
plt.show()

