#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:10:24 2021

@author: sollyboukman
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
from IPython.display import HTML
import plotly.graph_objects as go

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


one_hot_ready = one_hot_ready.to_numpy()

train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes:    the number of classes
        attr_dist:    a 2d NumPy array of the attribute distributions
        label_priors: a 1d NumPy array of the priors distribution
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None
        self.false_negs = 0
        self.false_pos = 0
        self.true_negs = 0
        self.true_pos = 0

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a n_examples x n_attributes numpy array
            y_train: a n_examples numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        self.false_negs = 0
        self.false_pos = 0
        self.true_negs = 0
        self.true_pos = 0
        #print(X_train)
        num_good_credit = np.sum(y_train)
        #print(num_good_credit)
        num_bad_credit = len(y_train) - num_good_credit
        #print(num_bad_credit)
        self.label_priors = np.array([(num_bad_credit + 1) / (len(y_train) + self.n_classes),
                                      (num_good_credit + 1) / (len(y_train) + self.n_classes)])
        
        num_attr = np.shape(X_train)[1]
        
        col_vec = np.array(list(map(lambda x: [x], y_train)))
        
        
        

        
        full_data = np.concatenate((X_train, col_vec), 1)
        
        
        #print(np.shape(full_data ))
        good_cs_rows = full_data[full_data[:,-1] == 1][:,:-1]
        #print(np.shape(good_cs_rows))
        bad_cs_rows = full_data[full_data[:,-1] == 0][:,:-1]
        

        
        att_dist = []
        
        for i in range(num_attr):
            
            num_with_good_credit = np.sum(good_cs_rows[:, i])
            #print(num_with_good_credit / (len(good_cs_rows) + self.n_classes))
            
            num_with_bad_credit = np.sum(bad_cs_rows[:, i])
            #print(num_with_bad_credit / (len(bad_cs_rows) + self.n_classes))
            att_dist.append(np.array([((num_with_bad_credit + 1) / (len(bad_cs_rows) + self.n_classes)),
                                      ((num_with_good_credit + 1) / (len(good_cs_rows) + self.n_classes))]))
        
        self.attr_dist = np.array(att_dist)
        
        
        
        return self.attr_dist, self.label_priors
    
    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.
            Remember to convert to log space to avoid overflow/underflow
            errors!

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a 1-D numpy array of predictions
        """
        
        
        predictions = []
        for input in inputs:
            
            col_vec = np.array(list(map(lambda x: [x], input)))
            
            full = np.concatenate((col_vec, self.attr_dist), 1)
            #print(np.shape(full))
            
            for i in range(len(input)):
                
                to_norm = np.sum(full[i,1:], axis=0)
                #print(type(to_norm))
                
                full[i,1:] = full[i,1:] / to_norm
                
                if full[i,0] == 1:
                    continue;
                else:
                    to_sub_from = np.ones(self.n_classes + 1)
                    full[i,:] = to_sub_from - full[i,:]
            
            #print(full[:,1:])
            #print(np.log(full[:,1:]))
            #print(np.shape(np.sum(np.log(full[:,1:]), axis=)))
            
            class_prediction = np.argmax(np.exp(np.sum(np.log(full[:,1:]), axis=0)))
            #print(class_prediction)
            predictions.append(class_prediction)
        
        #print(predictions)
        return np.array(predictions)

    def accuracy(self, X_test, y_test,train_run=True):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        
        predictions = self.predict(X_test)
        
        ret = np.mean(predictions == y_test)
        if not train_run:
            for i in range(len(y_test)):
                if predictions[i] == 1:
                    if y_test[i] == 1:
                        self.true_pos += 1
                    else:
                        self.false_pos += 1
                else:
                    if y_test[i] == 1:
                        self.false_negs += 1
                    else:
                        self.true_negs += 1
                
        
        return ret

#TESTING MODEL
nbc = NaiveBayes(2)

train_accuracies = []
test_accuracies = []


true_negs = []

true_pos = []

false_negs = []

false_pos = []

rand_accuracies = []

for i in range(100):
    
    nbc.train(train_x, train_y)
    
    train_accuracies.append(nbc.accuracy(train_x, train_y))
    
    test_accuracies.append(nbc.accuracy(test_x, test_y, train_run=False))
    rand_accuracies.append(np.sum(test_y == np.random.randint(2, size=np.shape(test_y))) / len(test_y))
    true_negs.append(nbc.true_negs)

    true_pos.append(nbc.true_pos)
    
    false_negs.append(nbc.false_negs)
    
    false_pos.append(nbc.false_pos)
    
    one_hot_ready = np.append(train_x, test_x, axis=0)
    encoded_labels = np.append(train_y, test_y)
    
    train_x, test_x, train_y, test_y = train_test_split(one_hot_ready, encoded_labels, test_size = 0.3,random_state=2, shuffle=True, stratify=encoded_labels)

    
print(sum(train_accuracies) / len(train_accuracies))

print(sum(test_accuracies) / len(test_accuracies))
#average train acc = 0.6257254901960786

# gen t-test stat
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

plt.plot(list(range(1,101)), train_accuracies)
plt.plot(list(range(1,101)), test_accuracies)
plt.plot(list(range(1, 101)), rand_accuracies)
plt.plot(list(range(1,101)), [sum(train_accuracies) / len(train_accuracies)]*100)
plt.plot(list(range(1,101)), [sum(test_accuracies) / len(test_accuracies)]*100)
plt.plot(list(range(1, 101)), [sum(rand_accuracies) / len(rand_accuracies)]*100)
plt.xlabel('Data Shuffle Iteration ')
plt.ylabel('Accuracies')
plt.title('Naive Bayes Accuracy Over Each Shuffle Iteration')
plt.legend(["Train Accuracies", "Test Accuracies", "Random Guessing Accuracies",
            "Mean Train Accuracy", "Mean Test Accuracy", "Mean Random Guess Accuracy"], loc='upper center', bbox_to_anchor=(0.5, -0.2)) 

#Test Accuracy 0.676829268292683

avg_true_negs = round(sum(true_negs) / len(true_negs))

avg_true_pos = round(sum(true_pos) / len(true_pos))

avg_false_negs = round(sum(false_negs) / len(false_negs))

avg_false_pos = round(sum(false_pos) / len(false_pos))

#print(avg_true_negs + avg_true_pos + avg_false_negs + avg_false_pos)

mat = np.array([[avg_true_negs, avg_false_negs], [avg_false_pos, avg_true_pos]])
'''
x = ['sleep together'] * avg_true_negs
y = ['sleep alone'] * avg_true_pos
w = ['sleep together'] * avg_false_negs
z = ['sleep alone'] * avg_false_pos

x1 = ['sleep together'] * avg_true_negs
y1 = ['sleep alone'] * avg_true_pos
w1 = ['sleep alone'] * avg_false_negs
z1 = ['sleep together'] * avg_false_pos
'''
#preds=x + y + w + z

#ytest=x1 + y1 + w1 + z1

#preds = np.array(preds)
#ytest = np.array(ytest)

#Code to plot confusion mat must be run separately
'''
sns.heatmap(m,annot=True, cmap="coolwarm" ,fmt='g')
plt.xticks([0.5,1.5],labels=np.array(['sleep alone', 'sleep together']))
plt.yticks([0,2],labels=np.array(['sleep alone', 'sleep together']))
plt.title('Naive Bayes Confusion matrix')
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
'''
