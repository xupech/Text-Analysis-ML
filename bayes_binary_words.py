#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:57:23 2019

@author: xupech
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

def word_binary(document):
    features = {}
    for word in document:
        features[word] = 1
    return features

"""
A helper function. Return a dictionary with words used featured as 1, not used as 0.
"""


allsets = [(word_binary(d), c) for (d,c) in documents]
featuresets =[feature[0] for feature in allsets]
labelsets = [feature[1] for feature in allsets]

"""
Get features and labels in allsets. Separate features and labels in different lists.
"""

v = DictVectorizer(sparse=False)
feature_ = v.fit_transform(featuresets)

le = preprocessing.LabelEncoder()
le.fit(labelsets)
label_=le.transform(labelsets)

X_train, X_test, y_train, y_test = train_test_split(
    feature_, label_, test_size=0.1, random_state=0)

bayes_binary_words = MultinomialNB()
bayes_binary_words.fit(X_train, y_train)

with open('/Users/xupech/Documents/GitHub/text-analysis-xupech/classifiers/bayes_binary_words.pkl', 'wb') as output:
    pickle.dump(bayes_binary_words, output, -1)
output.close()