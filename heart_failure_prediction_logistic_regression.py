# -*- coding: utf-8 -*-
"""Heart Failure Prediction logistic regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14o2z_Xa-2B1H0NsE7C0dkRbCUaAF9frL

# **importing section  reading dataset and printing basic info**
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('/content/heart_failure_clinical_records_dataset.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

df.nunique()

"""# **Pie Chart**"""

# prompt: pie chart dead event and no dead event

import matplotlib.pyplot as plt
# Get the number of dead and not dead events
dead_events = df['DEATH_EVENT'].value_counts()[1]
not_dead_events = df['DEATH_EVENT'].value_counts()[0]

# Create a pie chart
plt.pie([dead_events, not_dead_events], labels=['Dead', 'Not Dead'], autopct='%1.1f%%')
plt.title('Pie Chart of Dead and Not Dead Events')
plt.show()

# prompt: piechart for anaemia analysis


# Count the number of patients with and without anemia
anemia_count = df['anaemia'].value_counts()[1]
no_anemia_count = df['anaemia'].value_counts()[0]

# Create a pie chart
labels = ['Anemia', 'No Anemia']
sizes = [anemia_count, no_anemia_count]
colors = ['lightgreen', 'lightblue']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart of Anemia and No Anemia Cases')
plt.show()

# prompt: pie chart for diabetes analysis


# Count the number of patients with and without diabetes
diabetes_count = df['diabetes'].value_counts()[1]
no_diabetes_count = df['diabetes'].value_counts()[0]

# Create a pie chart
labels = ['Diabetes', 'No Diabetes']
sizes = [diabetes_count, no_diabetes_count]
colors = ['blue', 'pink']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart of Diabetes and No Diabetes Cases')
plt.show()

# prompt: pie chart for gender

import matplotlib.pyplot as plt
# Count the number of male and female patients
male_count = df['sex'].value_counts()[0]
female_count = df['sex'].value_counts()[1]

# Create a pie chart
labels = ['Male', 'Female']
sizes = [male_count, female_count]
colors = ['lavender', 'olive']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart of Gender')
plt.show()

# prompt: pie chart for high blood pressure

import matplotlib.pyplot as plt
# Count the number of patients with and without high blood pressure
high_bp_count = df['high_blood_pressure'].value_counts()[1]
no_high_bp_count = df['high_blood_pressure'].value_counts()[0]

# Create a pie chart
labels = ['High Blood Pressure', 'No High Blood Pressure']
sizes = [high_bp_count, no_high_bp_count]
colors = ['red', 'purple']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart of High Blood Pressure and No High Blood Pressure Cases')
plt.show()

# prompt: pie chart for smoking

import matplotlib.pyplot as plt
# Count the number of patients who smoke and don't smoke
smoking_count = df['smoking'].value_counts()[1]
no_smoking_count = df['smoking'].value_counts()[0]

# Create a pie chart
labels = ['Smoking', 'No Smoking']
sizes = [smoking_count, no_smoking_count]
colors = ['magenta', 'cyan']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart of Smoking and No Smoking Cases')
plt.show()

"""# **heat map and important features extract**"""

# prompt: heat map for df



# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# prompt: which is important features in dataset from heatmap make it as list

# Get the heatmap data
#heatmap_data = df.corr()

# Find the columns with the highest absolute correlation with the target variable
#important_features = heatmap_data['DEATH_EVENT'].sort_values(ascending=False)[1:8].index.tolist()

# Print the list of important features
#print(important_features)

"""# **modeling : logistic regression**"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

important_features=['time','ejection_fraction','serum_creatinine']
# Split the dataset into train and test sets
important_features
X = df[important_features]  # Features
y = df['DEATH_EVENT']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

reg= LogisticRegression()
reg.fit(X_train,y_train)

"""# **testing**"""

reg.predict(X_test)

accuracy_score(reg.predict(X_test),y_test)

"""# **confusion matrix ploting**"""

# prompt: confusion matrix ploting

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Predict the labels for the test set
y_pred = reg.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

import pickle
pickle.dump(reg,open('model.pkl','wb'))