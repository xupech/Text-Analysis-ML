"""main_3a.py

An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.

An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).

"""

import os
import nltk
import math
import re
from nltk.text import Text as text1
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


STOPLIST = set(nltk.corpus.stopwords.words())
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())


def read_text(path):
    if os.path.isfile(path) == True:
        txt = open(path, 'r').read()
    
    elif os.path.isdir(path) == True:
        filelists = PlaintextCorpusReader(path, '.*.mrg')
        txt = filelists.raw()
        
    return txt

def is_content_word(word):
    return word.lower() not in STOPLIST and word[0].isalpha()

class Text(text1):

    def __init__(self, path):
        raw = read_text(path)
        tokens = nltk.word_tokenize(raw)
        text1.__init__(self,tokens)
        self.t = tokens
        self.r = raw
        
        
    def token_count(self):
        return len(self.t)

    def type_count(self):
        return len(set([w.lower() for w in self.t]))

    def sentence_count(self):
        return len([t for t in self.t if t in ('.', '!', '?')])
    
    def most_frequent_content_words(self):
        dist = FreqDist([w for w in self.t if is_content_word(w)])
        return dist.most_common(n=25)
    
    def most_frequent_bigrams(self):
        filtered_bigrams = [b for b in list(nltk.bigrams(self.t))
                            if is_content_word(b[0]) and is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=25)
    
    def find_sirs(self):
        sirs = re.compile(r"[sS]ir \w*(?:-*\w*)*")
        m = sirs.findall(self.r)
        s = set(m)
        sm = list(s)
        return sm
    
    def find_brackets(self):
        brackets = re.compile(r"(?:\[|\().*?(?:\]|\))")
        # brackets = re.compile(r"(?:\[|\().+(?:\]|\))"), greedy v.s. nongreedy
        m = brackets.findall(self.r)
        s = set(m)
        sm = list(s)
        return sm
    
    def find_roles(self):
        rolesgrail = re.compile(r"\n(?!SCENE)(.+?):")
        # parathesis can be used to choose what content I would like to show.
        #rolesgrail = re.compile(r"^(?!SCENE)[^:]*", re.M)
        #rolesgrail = re.compile(r"(?<=\n)(?!SCENE)(?:[^:])*")
        #rolesgrail = re.compile(r"(?<=\n)(?!SCENE)(?:[^:])*")
        #rolesgrail = re.compile(r"^(?!SCENE)[^:]*")
        m = rolesgrail.findall(self.r)
        s = set(m)
        sm = list(s)
        return sm
    
    def find_repeated_words(self):
        repeated = re.compile(r"(\b\w\w\w+\b)\s?((\1))+")
        m = repeated.findall(self.r)
        s = set(m)
        sm = list(s)
        
        new_m = []
        for ele in sm:
            space = ' '
            new = str(space.join(ele))
            new_m.append(new)
        return new_m
    
    def apply_fsa(self,fsa):
        final_list = []
        tokens = self.t
        for ele in range(len(tokens)-2):
            w = tokens[ele]+' '+tokens[ele+1]
            if fsa.accept(w) == True:
                final_list.append((ele, w))
        return final_list

        
class Vocabulary():
    def __init__(self, Text):
        self.text = Text
        # keeping the unfiltered list around for statistics
        self.all_items = set([w.lower() for w in self.text])
        self.items = self.all_items.intersection(ENGLISH_VOCABULARY)
        # restricting the frequency dictionary to vocabulary items
        self.fdist = FreqDist(t.lower() for t in self.text if t.lower() in self.items)
        self.text_size = len(self.text)
        self.vocab_size = len(self.items)

    def __str__(self):
        return "<Vocabulary size=%d text_size=%d>" % (self.vocab_size, self.text_size)

    def __len__(self):
        return self.vocab_size

    def frequency(self, word):
        return self.fdist[word]

    def pos(self, word):
        # do not volunteer the pos for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        return synsets[0].pos() if synsets else 'n'

    def gloss(self, word):
        # do not volunteer the gloss (definition) for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        # make a difference between None for words not in vocabulary and words
        # in the vocabulary that do not have a gloss in WordNet
        return synsets[0].definition() if synsets else 'NO DEFINITION'

    def kwic(self, word):
        self.text.concordance(word)
        

""" Finite State Processing """

class FSA():
    def __init__(self, name, states, final_states, transitions):
        self.name = name
        self.states = states
        self.final_states = final_states
        self.transitions = transitions
        
    
    def pp(self):
        state =[]
        for transition in self.transitions:
            if transition[0] not in self.final_states:
                if transition[0] not in state:
                    print('<'+'State '+transition[0]+'>')
                
            else:
                print('<'+'State '+transition[0]+' f>')
            state.append(transition[0])   
            print (transition[1]+'-->'+transition[2])
        
        for ele in self.states:
            if ele not in state:
                if ele in self.final_states:
                    print('<'+'State '+ele+' f>')
                else:
                    print('<'+'State '+ele+'>')

    def accept(self, string):
        index = 0
        current_state = 'S0'
        future_state = None
        while index <= len(string):
            index_copy = index
            if index == len(string):
                if future_state == None:
                    return False

                if current_state in self.final_states:
                    return True
                else:
                    return False
            else:
                future_state = None
                for transition in self.transitions:
                    if current_state == transition[0] and string[index:index+len(transition[1])] == transition[1]:
                        future_state = transition[2]
                        index+=len(transition[1])
                        break

                if index == index_copy:
                    index+=1
                                
                if future_state == None:
                    pass
                else:
                    current_state = future_state
            
                    
            
            
