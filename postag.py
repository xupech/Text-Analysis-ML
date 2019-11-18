import re
import os
import math
import operator
import nltk
from nltk.corpus import brown
import pickle
from pickle import dump

brown_sents = brown.sents(categories = 'news')
brown_tagged_sents = brown.tagged_sents(categories='news')


t0=nltk.DefaultTagger('NN')
t1=nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
t2=nltk.BigramTagger(brown_tagged_sents, backoff=t1)

""" back trace model, first a bigram model, backoff to a unigram model, then backoff
    to a model with defaulttagger("NN")
"""

output=open('t2.pkl','wb')
pickle.dump(t2, output, -1)
output.close()