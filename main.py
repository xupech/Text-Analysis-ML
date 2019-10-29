"""main.py

Code scaffolding

"""

import os
import nltk
import math
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.corpus import stopwords

""" raw = open(path, 'r') doesn't work, read() command required to """ 
""" convert to string """
""" tokenize method recognizes only string input, this method removes all """ 
""" punctuations. Alternative: raw = open(path, 'r') """
"""        text = word for line in raw for word in line.lower().split() works """
"""        however punctuations remain."""

def read_text(path):
    if os.path.isfile(path) == True:
        raw = open(path, 'r').read()
        tokens = nltk.word_tokenize(raw)
        text = [token.lower() for token in tokens]
    elif os.path.isdir(path) == True:
        filelists = PlaintextCorpusReader(path, '.*')
        tokens = filelists.words()
        text = [token.lower() for token in tokens]
    return nltk.Text(text)
        # with open(path, 'r') as myfile:
          #  raw = myfile.read()
        # tokenizer=RegexpTokenizer(r'\w+')
        # tokens = tokenizer.tokenize(raw)
        # text = word.lower() for word in tokens

emma = read_text('/Users/xupech/Documents/GitHub/text-analysis-xupech/data/emma.txt')
grail = read_text('/Users/xupech/Documents/GitHub/text-analysis-xupech/data/grail.txt')
wsj = read_text('/Users/xupech/Documents/GitHub/text-analysis-xupech/data/wsj')

def token_count(text):
    return len(text)

def type_count(text):
    type_count = len(sorted(set(text)))-10
    return type_count


def sentence_count(text):
    count = 0
    freq = FreqDist(text)
    for key in freq:
        if key == '.' or key == '!' or key == '?':
            count += freq[key]
    return count


def most_frequent_content_words(text):
    p = list(string.punctuation)
    # punctuation list
    s = stopwords.words('english')
    # stopwords list
    freq = FreqDist(text)
    freq1 = freq.copy()
    for key in freq1:
        if key in s or key in p or key in ['--', "''", '``']:
            del freq[key]
    return freq.most_common(25)
    

def most_frequent_bigrams(text):
    p = list(string.punctuation)
    # punctuation list
    s = stopwords.words('english')
    # stopwords list
    bgs = list(nltk.bigrams(text))
    freq = nltk.FreqDist(bgs)
    freq1 = freq.copy()
    for key in freq1:
        if key[0] in s or key[0] in p or key[0] in ['--', "''", '``']:
            del freq[key]
        elif key[1] in s or key[1] in p or key[1] in ['--', "''", '``']:
            del freq[key]
    return freq.most_common(25)

class Vocabulary():

    def __init__(self, text):
        self.txt = text

    def frequency(self, word):
        text1 = self.txt
        freq = FreqDist(text1)
        return freq.get(word, 0)

    def pos(self, word):
        if word in self.txt:
            pos = nltk.tag.pos_tag([word])
            return (pos[0][1])
        else:
            print (None)

    def gloss(self, word):
        synset = wn.synsets(word)[0]
        gloss = synset.definition
        return gloss

    def quick(self, word):
        text1 = self.txt
        kwic = text1.concordance(word)
        return kwic

categories = ('adventure', 'fiction', 'government', 'humor', 'news')


def compare_to_brown(text):
    s = stopwords.words('english')
    p = list(string.punctuation)
    a = ['--', "''", '``']
    
    txt = [t for t in text if not t in s or not t in p or not t in a]
    txtset = {w for w in txt}
    freqtxt = FreqDist(txt)
    t = dict(sorted(freqtxt.items()))
    
    for ele in categories:
        cat = brown.words(categories=ele)
   
        cat2=[t.lower() for t in cat if not t in 
                   s or not t in p or not t in a]
        
        brownset={w for w in cat2}
        freqbrown = FreqDist(cat2)
        f = dict(sorted(freqbrown.items()))        
    
        v1=txtset.union(brownset)                         
        v = sorted(v1)
        #union set
    
        l1 =[];l2 =[]
        for m in v:
            if m in t: 
                l1.append(t.get(m))
            else:
                l1.append(0)
            if m in f: 
                l2.append(f.get(m))
            else:
                l2.append(0)


        k = 0
        x = 0
        y = 0
        for i in range(len(v)):
            k+= l1[i]*l2[i]
            x+= ((l1[i])**2)
            y+= ((l2[i])**2)
        cosine = round(k/(math.sqrt(x)*math.sqrt(y)),2)
        print( ele + '     ' + str(cosine))

if __name__ == '__main__':

    text = read_text('data/grail.txt')
    token_count(text)
