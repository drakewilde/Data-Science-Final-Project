#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:39:30 2021

@author: sollyboukman
"""
import pandas as pd

sleeping_alone_data = pd.read_csv('sleeping_alone_data.csv')
#sleeping_alone_data.info()

all_atts =['Gender',
               'Household Income',
               'Age',
               'Education',
               'Which of the following best describes your current occupation?',
               'Which of the following best describes your current relationship status?',
               'How long have you been in your current relationship? If you are not currently in a relationship, please answer according to your last relationship.',
               'When both you and your partner are at home, how often do you sleep in separate beds?',
               'Location (Census Region)']
print('Data Sample')
for i in range(10):
    print(sleeping_alone_data[all_atts].sample(n=1, axis=0))

#sleeping_alone_data[all_atts].head()


#These can be uncommented one by one to plot. This checks the distribution of values for relevant attributes
'''
metric = sleeping_alone_data['Gender'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Household Income'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Age'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Education'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Which of the following best describes your current occupation?'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Which of the following best describes your current relationship status?'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['How long have you been in your current relationship? If you are not currently in a relationship, please answer according to your last relationship.'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['When both you and your partner are at home, how often do you sleep in separate beds?'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')

metric = sleeping_alone_data['Location (Census Region)'].astype(str).value_counts()
if 'Response' in metric:
    del metric['Response']
metric.plot(kind='bar')
'''
