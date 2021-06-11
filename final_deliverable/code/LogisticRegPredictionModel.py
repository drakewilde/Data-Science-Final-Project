#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:26:50 2021
@author: Shady
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.discrete.discrete_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from random import randint
import seaborn as sns; sns.set()

data = pd.read_csv('../data_deliverable/data/sleeping_alone_data.csv').drop(0)
data['StartDate'] = pd.to_datetime(data['StartDate'], errors='coerce')
data['EndDate'] = pd.to_datetime(data['EndDate'], errors='coerce')


data['TimeTaken'] = data['EndDate'] - data['StartDate']
data['TimeTaken'] = data['TimeTaken'].dt.total_seconds().div(60).astype(int)


q1 = data['TimeTaken'].quantile(0.25)
q2 = data['TimeTaken'].quantile(0.75)

bound = 1.5 * (q2 - q1)

#data = data[data['TimeTaken'] <= (q2 + bound)]
#data = data[data['TimeTaken'] >= (q1 - bound)]
#'Which of the following best describes your current occupation?',
all_atts =['Gender',
               'Household Income',
               'Age',
               'Education',
               'Which of the following best describes your current relationship status?',
               'How long have you been in your current relationship? If you are not currently in a relationship, please answer according to your last relationship.',
               'When both you and your partner are at home, how often do you sleep in separate beds?',
               'Location (Census Region)']
#'TimeTaken']

data2 = data[all_atts].dropna()

relevant_attributes = data2
#dummy vars

gender_dummys = pd.get_dummies(relevant_attributes['Gender'], prefix='Gender',dummy_na=False, drop_first=True)

income_dummys = pd.get_dummies(relevant_attributes['Household Income'], prefix='Income',dummy_na=False, drop_first=True)

age_dummys = pd.get_dummies(relevant_attributes['Age'], prefix='Age',dummy_na=False, drop_first=True)

education_dummys = pd.get_dummies(relevant_attributes['Education'], prefix='Education',dummy_na=False, drop_first=True)

#occupation_dummys = pd.get_dummies(relevant_attributes['Which of the following best describes your current occupation?'], prefix='Occupation',dummy_na=False, drop_first=True)

relationship_status_dummys = pd.get_dummies(relevant_attributes['Which of the following best describes your current relationship status?'], prefix='Relationship_Status', dummy_na=False, drop_first=True)

relationship_length_dummys = pd.get_dummies(relevant_attributes['How long have you been in your current relationship? If you are not currently in a relationship, please answer according to your last relationship.'], prefix='Relationship_Length',dummy_na=False, drop_first=True)

location_dummys = pd.get_dummies(relevant_attributes['Location (Census Region)'], prefix='Location',dummy_na=False, drop_first=True)

labels = np.array(relevant_attributes['When both you and your partner are at home, how often do you sleep in separate beds?'].to_list())

encoded_labels = np.zeros(len(labels))

for i in range(len(labels)):
    if labels[i] == 'Never' or labels[i] == 'Once a year or less':
        continue
    else:
        encoded_labels[i] = 1
  
one_hot_ready = pd.concat([gender_dummys, income_dummys, age_dummys, education_dummys, relationship_status_dummys, relationship_length_dummys, location_dummys], axis=1)

one_hot_ready = one_hot_ready.to_numpy()

train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)
#controls = pd.concat([income_dummys, age_dummys, relationship_status_dummys, relationship_length_dummys], axis=1)
#one_hot_ready['agesx'] = encoded_age.squeeze().tolist()


print('Using all relevant atts')

logistic_regression = LogisticRegression()

#one_hot_ready = pd.concat([gender_dummys, income_dummys, age_dummys, education_dummys, relationship_status_dummys, relationship_length_dummys, location_dummys], axis=1)

#location_dummys = location_dummys.to_numpy()

#train_x, test_x, train_y, test_y = train_test_split(location_dummys, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)


train_accuracies = []
test_accuracies = []
rand_accuracies = []

m = np.array([[0,0],[0,0]])


#random_controls = np.random.randint(2, size=np.shape(one_hot_ready))
#train_x, test_x, train_y, test_y = train_test_split(random_controls, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)

for i in range(100):
    
    logistic_regression.fit(train_x, train_y)
    train_accuracies.append(metrics.accuracy_score(train_y, logistic_regression.predict(train_x)))
    rand_accuracies.append(np.sum(test_y == np.random.randint(2, size=np.shape(test_y))) / len(test_y))
    test_accuracies.append(metrics.accuracy_score(test_y, logistic_regression.predict(test_x)))
    
    m += confusion_matrix(test_y, logistic_regression.predict(test_x))
    
    one_hot_ready = np.append(train_x, test_x, axis=0)
    encoded_labels = np.append(train_y, test_y)
    
    train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)




print(sum(train_accuracies) / len(train_accuracies))

print(sum(test_accuracies) / len(test_accuracies))

testmean = np.mean(test_accuracies)
testvar = np.var(test_accuracies)

randmean = np.mean(rand_accuracies)
randvar = np.var(rand_accuracies)

t_stat = ((testmean - randmean)/(((testvar/len(test_accuracies)) + (randvar/len(rand_accuracies))) ** 0.5))
print(t_stat)
standarderr = ((testvar/len(test_accuracies)) + (randvar/len(rand_accuracies))) ** 0.5
print(f'Standard Error {((testvar/len(test_accuracies)) + (randvar/len(rand_accuracies))) ** 0.5}')
tstar = 1.9842
lower_bound = testmean - tstar * standarderr
upper_bound = testmean + tstar * standarderr

plt.plot(list(range(1, 101)), train_accuracies)
plt.plot(list(range(1, 101)), test_accuracies)
plt.plot(list(range(1, 101)), rand_accuracies)
plt.plot(list(range(1, 101)), [
         sum(train_accuracies) / len(train_accuracies)]*100)
plt.plot(list(range(1, 101)), [sum(test_accuracies) / len(test_accuracies)]*100)
plt.plot(list(range(1, 101)), [sum(rand_accuracies) / len(rand_accuracies)]*100)
plt.xlabel('Data Shuffle Iteration ')
plt.ylabel('Accuracies')
plt.title('Logistic Reg. Accuracy Over Each Shuffle Iteration')
plt.legend(["Train Accuracies", "Test Accuracies", "Random Guessing Accuracies",
            "Mean Train Accuracy", "Mean Test Accuracy", "Mean Random Guess Accuracy"], loc='upper center', bbox_to_anchor=(0.5, -0.2))
#Accuracy of income: 74.01
#accuracy of age same
#accuracy of relation status 
preds = logistic_regression.predict(test_x)
m = np.round(m/100)
'''
sns.heatmap(m,annot=True, cmap="coolwarm" ,fmt='g')
plt.xticks([0.5,1.5],labels=np.array(['sleep alone', 'sleep together']))
plt.yticks([0,2],labels=names)
plt.title('Confusion matrix')
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
'''
