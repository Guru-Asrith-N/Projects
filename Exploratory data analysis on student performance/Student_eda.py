import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("student-mat.csv",sep=';')

print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.isnull().sum())

print(df.describe())

print(df['G3'].mean())

print(df.groupby('sex')['G3'].mean())

print(df.groupby('studytime')['G3'].mean())

print(df.groupby('failures')['G3'].mean())

sns.histplot(df['G3'],kde=True)
plt.title("Distribution of Final grdes")
plt.show()

sns.boxplot(x='sex',y='G3',data = df)
plt.title("Final grades by gender")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()


