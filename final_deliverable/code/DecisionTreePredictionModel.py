#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import graphviz

#Data Preprocessing

sleeping_alone_data = pd.read_csv('../data_deliverable/data/sleeping_alone_data.csv')

all_atts =['Gender',
               'Household Income',
               'Age',
               'Education',
               'Which of the following best describes your current relationship status?',
               'How long have you been in your current relationship? If you are not currently in a relationship, please answer according to your last relationship.',
               'When both you and your partner are at home, how often do you sleep in separate beds?',
               'Location (Census Region)']

relevant_attributes = sleeping_alone_data[all_atts].dropna()


relevant_attributes = relevant_attributes.drop(0)

gender_dummys = pd.get_dummies(relevant_attributes['Gender'], prefix='Gender',dummy_na=False, drop_first=True)

income_dummys = pd.get_dummies(relevant_attributes['Household Income'], prefix='Income',dummy_na=False, drop_first=True)

age_dummys = pd.get_dummies(relevant_attributes['Age'], prefix='Age',dummy_na=False, drop_first=True)

education_dummys = pd.get_dummies(relevant_attributes['Education'], prefix='Education',dummy_na=False, drop_first=True)

#occupation_dummys = pd.get_dummies(relevant_attributes['Which of the following best describes your current occupation?'], prefix='Occupation',dummy_na=True)

relationship_status_dummys = pd.get_dummies(relevant_attributes['Which of the following best describes your current relationship status?'], prefix='Relationship_Status',dummy_na=False, drop_first=True)

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

#one_hot_ready['labels'] = encoded_labels 
features = one_hot_ready.columns

one_hot_ready = one_hot_ready.to_numpy()

train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)


# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth=3)


train_accuracies = []
test_accuracies = []

rand_accuracies = []
m = np.array([[0,0],[0,0]])

for i in range(100):
    
    clf = clf.fit(train_x,train_y)
    
    train_accuracies.append(metrics.accuracy_score(train_y, clf.predict(train_x)))
    
    rand_accuracies.append(np.sum(test_y == np.random.randint(2, size=np.shape(test_y))) / len(test_y))
    
    test_accuracies.append(metrics.accuracy_score(test_y, clf.predict(test_x)))
    m += confusion_matrix(test_y, clf.predict(test_x))
    one_hot_ready = np.append(train_x, test_x, axis=0)
    encoded_labels = np.append(train_y, test_y)
    
    train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)

# t-test stat

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
# Train Decision Tree Classifer


#Predict the response for test dataset

'''
y_pred = clf.predict(test_x)

## Model Accuracy
print("Accuracy:", metrics.accuracy_score(test_y, y_pred))
'''
'''
print(sum(train_accuracies) / len(train_accuracies))

print(sum(test_accuracies) / len(test_accuracies))

#Visualize Data
plt.plot(list(range(1,101)), train_accuracies)
plt.plot(list(range(1,101)), test_accuracies)
plt.plot(list(range(1, 101)), rand_accuracies)
plt.plot(list(range(1,101)), [sum(train_accuracies) / len(train_accuracies)]*100)
plt.plot(list(range(1,101)), [sum(test_accuracies) / len(test_accuracies)]*100)
plt.plot(list(range(1, 101)), [sum(rand_accuracies) / len(rand_accuracies)]*100)
plt.xlabel('Data Shuffle Iteration ')
plt.ylabel('Accuracies')
plt.title('Decision Tree Accuracy Over Each Shuffle Iteration')
plt.legend(["Train Accuracies", "Test Accuracies", "Random Guessing Accuracies",
            "Mean Train Accuracy", "Mean Test Accuracy", "Mean Random Guess Accuracy"], loc='upper center', bbox_to_anchor=(0.5, -0.2))#loc='center left', bbox_to_anchor=(1, 0.5))

m = np.round(m/100)
'''
'''
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=features,  
                                class_names=np.array(['sleep alone', 'sleep together']),
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 

'''

plt.figure(figsize=(10,5))

tree.plot_tree(clf, feature_names=features,  
                                class_names=np.array(['sleep alone', 'sleep together']), filled=True, fontsize=7)
plt.show()

'''
accuracyforDepth = []


for i in range(1, 11):
    
    clf1 = DecisionTreeClassifier(max_depth=i)
    clf1 = clf1.fit(train_x, train_y)
    accuracyforDepth.append(metrics.accuracy_score(test_y, clf1.predict(test_x)))
    one_hot_ready = np.append(train_x, test_x, axis=0)
    encoded_labels = np.append(train_y, test_y)
    
    train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)

plt.plot(list(range(1,11)), accuracyforDepth)
plt.xlabel('Max Depth')
plt.ylabel('Accuracies')
plt.title('Accuracy Over Each Decision Tree Depth')
plt.show()
'''
'''
sns.heatmap(m,annot=True, cmap="coolwarm" ,fmt='g')
plt.xticks([0.5,1.5],labels=np.array(['sleep alone', 'sleep together']))
plt.yticks([0,2],labels=np.array(['sleep alone', 'sleep together']))
plt.title('Decision Tree Confusion matrix')
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
'''