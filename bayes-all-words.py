#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:30:57 2019

@author: xupech
"""
"""
This is BayesMultinomialNB model for raw counts.
"""

import sys
import argparse
import re
import os
import math
import operator
import random
import nltk
import pickle
from pickle import load
from pickle import dump
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

"""
import movie_reviews into a list called documents. The list contains a tuple for words
in each review and label for the review.
"""

random.shuffle(documents)

"""
shuffle the documents.
"""

STOPLIST = set(nltk.corpus.stopwords.words())
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w not in STOPLIST)
all_words = defaultdict(lambda:0, all_words)

"""
Have all words and their frequencies stored in all_words. Remove stopwords. Have all words stored in word_features.
Set default value of dictionary = 0 in case the word is not in the dictionary.
"""

def raw_counts(document):
    features = {}
    for word in document:
        features[word]=all_words[word.lower()]
    return features
"""
A helper function.
return a dictionary, in which keys are words in word_features and their corresponding frequencies in
all reviews. Stop words will have frequency = 0.
"""

allsets = [(raw_counts(d), c) for (d,c) in documents]
featuresets =[feature[0] for feature in allsets]
labelsets = [feature[1] for feature in allsets]

"""
Get words frequency and labels in allsets. Separate features and labels in different lists.
"""

v = DictVectorizer(sparse=False)
feature_ = v.fit_transform(featuresets)

"""
Encode features.
"""

le = preprocessing.LabelEncoder()
le.fit(labelsets)
label_=le.transform(labelsets)
"""
Encode labels.
"""

X_train, X_test, y_train, y_test = train_test_split(
    feature_, label_, test_size=0.1, random_state=0)

"""
Separate training/test data by ratio 0.1/0.9.
"""

bayes_all_words = MultinomialNB()
bayes_all_words.fit(X_train, y_train)

"""
Init Naive Bayes MultinomialNB() model and fit the data.
"""

with open('/Users/xupech/Documents/GitHub/text-analysis-xupech/classifiers/bayes-all-words.pkl', 'wb') as output:
    pickle.dump(bayes_all_words, output, -1)
output.close()
"""
Model stored in a pkl file in classifiers folder. Relative direction somehow doesn't work. So I use absolute directions.
"""