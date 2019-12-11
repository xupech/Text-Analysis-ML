#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:03:35 2019

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
from sklearn.tree import DecisionTreeClassifier
import timeit


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

MPQA = {}
subword = re.compile(r"word1=([^\s]+)")
subj = re.compile(r"priorpolarity=([^\n]+)")
with open('subjclueslen1-HLTEMNLP05.tff') as r:
   for line in r:
       MPQA[subword.findall(line)[0]]=subj.findall(line)[0]           
"""
Import subjectivity lexicon words file and use regular expression to recognize lexicon words in the file.
Store words in dictionary with positive/negative as values.
"""

negation_doc = documents[:]
negation_words=[]
for document in negation_doc:
    d = document[0]
    for index in range(len(d)):
        if d[index] in ['not',"n't",'hardly','never', 'no', "didn't", "can't", "don't", "doesn't",
                    "aren't", "weren't", "wasn't", "couldn't"]:
            n = 0
            while index+n< len(d) and d[index+n].isalpha():
                d[index+n]='not_'+d[index+n]
                n+=1
        else:
            continue
    negation_words+=d
"""
Create a new dataset with 'not_' adding before a word that comes after negation words and before 
punctuation. Collect all words into a list. 
"""                
all_words_neg = nltk.FreqDist(w.lower() for w in negation_words  if w not in STOPLIST)
all_words_neg = defaultdict(lambda:0, all_words_neg)    
"""
Get a frequency distribution of all words plus negation status words using freqdist. More words are generated
this time with more than 40k, compared to 36k in original dataset.
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

def word_binary(document):
    features = {}
    for word in document:
        features[word] = 1
    return features
"""
A helper function. Return a dictionary with words used featured as 1, not used as 0.
"""

def SWN_(document):
    features = {}
    for word in document:
        senti = list(swn.senti_synsets(word, 'a'))
        if len(senti)>0:
            if senti[0].pos_score() > 0.5:
                features[word] = 1
            elif senti[0].neg_score() > 0.5:
                features[word] = 0.5
            else:
                pass
    return features

"""
Feature helper function. Returns 1 for word with positive score>0.5, 0.5 with negative score > 0.5.
"""

def MPQA_(document):
    features = {}
    for word in document:
        if word in MPQA:
            if MPQA[word] == 'positive':
                features[word] = 1
            elif MPQA[word] == 'negative':
                features[word] = 2
        else:
            pass
    return features
"""
Help function. Feature word in document if word is in the lexicon words dictionary. Label word 1 if 
it is positive, label it 2 if it is negative.
"""

def neg_raw_counts(document):
    features = {}
    for word in document:
        features[word] = all_words_neg[word.lower()]
    return features

"""
Raw count all words, including negation words.
"""

def TRAIN():
    """
    1.
    Bayes all words model.
    """
    start = timeit.default_timer()
    
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

    X_train, X_test, y_train, y_test = train_test_split(
        feature_, labelsets, test_size=0.1, random_state=0)
    """
    Separate training/test data by ratio 0.1/0.9.
    """
    bayes_all_words = MultinomialNB()
    bayes_all_words.fit(X_train, y_train)
    """
    Init Naive Bayes MultinomialNB() model and fit the data.
    """
    print('Creating Bayes classifier in classifiers/bayes-all-words.pkl')
    print('Accuracy: {:.2f}'.format(bayes_all_words.score(X_test, y_test)))
    with open('./classifiers/bayes-all-words.pkl', 'wb') as output:
        pickle.dump(bayes_all_words, output, -1)
    output.close()
    """
    Model stored in a pkl file in classifiers folder. Relative direction somehow doesn't work. So I use absolute directions.
    """


    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))


    """
    2.
    Decision tree all words model.
    """
    start = timeit.default_timer()
    
    tree_all_words = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree_all_words.fit(X_train, y_train)   
    """
    Init Decision Tree model and fit the data.
    """ 
    with open('./classifiers/tree-all-words.pkl', 'wb') as output:
        pickle.dump(tree_all_words, output, -1)
    output.close()
    """
    Model stored in a pkl file in classifiers folder. Relative direction somehow doesn't work. So I use absolute directions.
    """
    print('Creating Tree classifier in classifiers/tree-all-words.pkl')
    print('Accuracy: {:.2f}'.format(tree_all_words.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
        
    """
    3.
    Bayes binary model.
    """
    start = timeit.default_timer()
    
    allsets = [(word_binary(d), c) for (d,c) in documents]
    featuresets =[feature[0] for feature in allsets]
    labelsets = [feature[1] for feature in allsets]    
    """
    Get features and labels in allsets. Separate features and labels in different lists.
    """  
    v = DictVectorizer(sparse=False)
    feature_ = v.fit_transform(featuresets)   

    X_train, X_test, y_train, y_test = train_test_split(
        feature_, labelsets, test_size=0.1, random_state=0)    
    bayes_binary_words = MultinomialNB()
    bayes_binary_words.fit(X_train, y_train)    
    with open('./classifiers/bayes_binary_words.pkl', 'wb') as output:
        pickle.dump(bayes_binary_words, output, -1)
    output.close()
    print('Creating Bayes classifier in classifiers/bayes-binary-words.pkl')    
    print('Accuracy: {:.2f}'.format(bayes_binary_words.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
    
    """
    4.
    Decision tree binary model.
    """
    start = timeit.default_timer()
    
    tree_binary = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree_binary.fit(X_train, y_train)    
    with open('./classifiers/tree_binary_words.pkl', 'wb') as output:
        pickle.dump(tree_binary, output, -1)
    output.close()  
    print('Creating Tree classifier in classifiers/tree-binary-words.pkl')
    print('Accuracy: {:.2f}'.format(tree_binary.score(X_test, y_test)))    
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
        
    """
    5.
    Bayes SentiWordNet Model.
    """    
    start = timeit.default_timer()

    allsets = [(SWN_(d), c) for (d,c) in documents]
    featuresets =[feature[0] for feature in allsets]
    labelsets = [feature[1] for feature in allsets]    
    v = DictVectorizer(sparse=False)
    feature_ = v.fit_transform(featuresets)   
   
    X_train, X_test, y_train, y_test = train_test_split(
        feature_, labelsets, test_size=0.1, random_state=0)    
    bayes_sentiwordnet = MultinomialNB()
    bayes_sentiwordnet.fit(X_train, y_train)   
    with open('./classifiers/bayes_sentiwordnet.pkl', 'wb') as output:
        pickle.dump(bayes_sentiwordnet, output, -1)
    output.close()
    print('Creating Bayes classifier in classifiers/bayes_sentiwordnets.pkl')    
    print('Accuracy: {:.2f}'.format(bayes_sentiwordnet.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
       
    """
    6.
    Decision tree SentiWordNet Model.
    """    
    start = timeit.default_timer()
    
    tree_sentiwordnet = DecisionTreeClassifier(random_state=0)
    tree_sentiwordnet.fit(X_train, y_train)
    with open('./classifiers/tree_sentiwordnet.pkl', 'wb') as output:
        pickle.dump(tree_sentiwordnet, output, -1)
    output.close()
    print('Creating Tree classifier in classifiers/tree_sentiwordnets.pkl')        
    print('Accuracy: {:.2f}'.format(tree_sentiwordnet.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
       
    """
    7. 
    Subjectivity lexicon words bayes model.
    """    
    start = timeit.default_timer()
        
    allsets = [(MPQA_(d), c) for (d,c) in documents]
    featuresets =[feature[0] for feature in allsets]
    labelsets = [feature[1] for feature in allsets]    
    v = DictVectorizer(sparse=False)
    feature_ = v.fit_transform(featuresets)    
   
    X_train, X_test, y_train, y_test = train_test_split(
        feature_, labelsets, test_size=0.1, random_state=0)    
    bayes_subjectivity = MultinomialNB()
    bayes_subjectivity.fit(X_train, y_train)    
    with open('./classifiers/bayes_MPQA.pkl', 'wb') as output:
        pickle.dump(bayes_subjectivity, output, -1)
    output.close()
    print('Creating Bayes classifier in classifiers/bayes_MPQA.pkl')    
    print('Accuracy: {:.2f}'.format(bayes_subjectivity.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
        
    """
    8. 
    Subjectivity decision tree model.
    """    
    start = timeit.default_timer()
    
    tree_subjectivity = DecisionTreeClassifier(random_state=0)
    tree_subjectivity.fit(X_train, y_train)    
    with open('./classifiers/tree_MPQA.pkl', 'wb') as output:
        pickle.dump(tree_subjectivity, output, -1)
    output.close()
    print('Creating tree classifier in classifiers/tree_MPQA.pkl')    
    print('Accuracy: {:.2f}'.format(tree_subjectivity.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
        
    
    """
    9.
    Negation all words naive bayes model.
    """    
    start = timeit.default_timer()
    
    allsets = [(neg_raw_counts(d), c) for (d,c) in negation_doc]
    featuresets =[feature[0] for feature in allsets]
    labelsets = [feature[1] for feature in allsets]   
    v = DictVectorizer(sparse=False)
    feature_ = v.fit_transform(featuresets)    
    
    X_train, X_test, y_train, y_test = train_test_split(
        feature_, labelsets, test_size=0.1, random_state=0)    
    bayes_negation_all_words = MultinomialNB()
    bayes_negation_all_words.fit(X_train, y_train)   
    with open('./classifiers/bayes_negation.pkl', 'wb') as output:
        pickle.dump(bayes_negation_all_words, output, -1)
    output.close()    
    print('Creating Bayes classifier in classifiers/bayes_negation.pkl')
    print('Accuracy: {:.2f}'.format(bayes_negation_all_words.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
        
    """
    10.
    Negation all words decision tree model.
    """    
    start = timeit.default_timer()    
    tree_negation = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree_negation.fit(X_train, y_train)    
    with open('./classifiers/tree_negation.pkl', 'wb') as output:
        pickle.dump(tree_negation, output, -1)
    output.close()
    print('Creating Tree classifier in classifiers/tree_negation.pkl')    
    print('Accuracy: {:.2f}'.format(tree_negation.score(X_test, y_test)))
    
    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    


"""
Prediction functions start here.
"""
def bayes_all_words(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = raw_counts(word)
    
    allsets = [(raw_counts(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)

    feature_new = v.transform(feature)
    input = open('./classifiers/bayes-all-words.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])    

def tree_all_words(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = raw_counts(word)
    
    allsets = [(raw_counts(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)

    feature_new = v.transform(feature)
    input = open('./classifiers/tree-all-words.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])

def bayes_binary_words(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = word_binary(word)
    
    allsets = [(word_binary(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)
    input=open('./classifiers/bayes_binary_words.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])
    
def tree_binary_words(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = word_binary(word)
    
    allsets = [(word_binary(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)
    input=open('./classifiers/tree_binary_words.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])

def bayes_sentiwordnet(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = SWN_(word)
    
    allsets = [(SWN_(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open('./classifiers/bayes_sentiwordnet.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])
    
def tree_sentiwordnet(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = SWN_(word)
    
    allsets = [(SWN_(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open('./classifiers/tree_sentiwordnet.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])
 
def bayes_MPQA(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = MPQA_(word)
    
    allsets = [(MPQA_(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open('./classifiers/bayes_MPQA.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])    

def tree_MPQA(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = MPQA_(word)
    
    allsets = [(MPQA_(d), c) for (d,c) in documents]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open('./classifiers/tree_MPQA.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])

def bayes_negation(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = neg_raw_counts(word)
    
    allsets = [(neg_raw_counts(d), c) for (d,c) in negation_doc]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open    ('./classifiers/bayes_negation.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])
    
def tree_negation(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = nltk.word_tokenize(text)
    feature = neg_raw_counts(word)
    

    allsets = [(neg_raw_counts(d), c) for (d,c) in negation_doc]
    featuresets =[f[0] for f in allsets]  
    v = DictVectorizer(sparse=False)
    v.fit_transform(featuresets)
    
    feature_new = v.transform(feature)    
    input=open    ('./classifiers/tree_negation.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_new)[0])
    
def Return_Result(model, index, text_path):
    if model == 'bayes' and index == '1':
        bayes_all_words(text_path)
    if model == 'tree' and index == '1':
        tree_all_words(text_path)
    if model == 'bayes' and index == '2':
        bayes_binary_words(text_path)
    if model == 'tree' and index == '2':
        tree_binary_words(text_path)
    if model == 'bayes' and index == '3':
        bayes_sentiwordnet(text_path)
    if model=='tree' and index == '3':
        tree_sentiwordnet(text_path)
    if model == 'bayes' and index == '4':
        bayes_MPQA(text_path)
    if model == 'tree' and index == '4':
        tree_MPQA(text_path)
    if model == 'bayes' and index == '5':
        bayes_negation(text_path)
    if model == 'tree' and index == '5':
        tree_negation(text_path)


parser = argparse.ArgumentParser(prog = 'classifier')
parser.add_argument('--train', dest = 'train', action = 'store_true')
parser.add_argument('--run', dest = 'run', type = str, nargs = '+')
args = parser.parse_args()

if args.train:
    TRAIN()
    
if args.run:
    path = './' + args.run[1]
    if len(args.run) == 2:
        print("""Choose a model:
1 - all words raw counts
2 - all words binary
3 - SentiWordNet words
4 - Subjectivity Lexicon words
5 - all words plus Negation""")
        model_index = input('Type a number:\n')         
        Return_Result(args.run[0], model_index, path)