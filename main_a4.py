import re
import os
import math
import operator
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader


# NLTK stoplist with 3136 words (multilingual)
STOPLIST = set(nltk.corpus.stopwords.words())

# Vocabulary with 234,377 English words from NLTK
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())


def is_content_word(word):
    """A content word is not on the stoplist and its first character is a letter."""
    return word.lower() not in STOPLIST and word[0].isalpha()

def find_plural_pattern(word):
    plural=re.compile(r"\b(\w+)s\b")
    return bool(plural.match(word))
    """
    In this case, we simply assume any word ends with 's' is in plural form.
    """


class Text(object):
    
    def __init__(self, path, name=None):
        """Takes a file path, which is assumed to point to a file or a directory, 
        extracts and stores the raw text and also stores an instance of nltk.text.Text."""
        self.name = name
        if os.path.isfile(path):
            self.raw = open(path).read()
        elif os.path.isdir(path):
            corpus = PlaintextCorpusReader(path, '.*.mrg')
            self.raw = corpus.raw()
        self.text = nltk.text.Text(nltk.word_tokenize(self.raw))
        self.tagged_text=nltk.pos_tag(self.text)


    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]

    def __str__(self):
        name = '' if self.name is None else " '%s'" % self.name 
        return "<Text%s tokens=%s>" % (name, len(self))

    def token_count(self):
        """Just return the length of the text."""
        return len(self)

    def type_count(self):
        """Returns the type count, with minimal normalization by lower casing."""
        # an alternative would be to use the method nltk.text.Text.vocab()
        return len(set([w.lower() for w in self.text]))

    def sentence_count(self):
        """Return number of sentences, using the simplistic measure of counting period,
        exclamation marks and question marks."""
        # could also use nltk.sent.tokenize on self.raw
        return len([t for t in self.text if t in '.!?'])

    def most_frequent_content_words(self):
        """Return a list with the 25 most frequent content words and their
        frequencies. The list has (word, frequency) pairs and is ordered
        on the frequency."""
        dist = nltk.FreqDist([w for w in self.text if is_content_word(w.lower())])
        return dist.most_common(n=25)

    def most_frequent_bigrams(self, n=25):
        """Return a list with the 25 most frequent bigrams that only contain
        content words. The list returned should have pairs where the first
        element in the pair is the bigram and the second the frequency, as in
        ((word1, word2), frequency), these should be ordered on frequency."""
        filtered_bigrams = [b for b in list(nltk.bigrams(self.text))
                            if is_content_word(b[0]) and is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=n)

    def concordance(self, word):
        self.text.concordance(word)
        

    def plural_words(self):
        text_plural={}
        for w in self.tagged_text:
            w_plur=w[0]
            if find_plural_pattern(w_plur):
                if w[0] in text_plural.keys():
                    if w[1]=='NNS' or w[1]=='NNPS':
                        text_plural[w_plur]+=1
                if w[0] not in text_plural.keys():
                    if w[1]=='NNS' or w[1]=='NNPS':
                        text_plural.update({w_plur:1})
        return text_plural

    """
    help function for nouns_more_common_in_plural_form()
    plural_word returns a dictionary as a local variable, key is plural words, value is frequency of 
    the plural words in text.
    """    
    

    def singular_words(self):
        plural = self.plural_words()
        text_singular={}
        for w in self.tagged_text:
            if w[1]=='NN' or w[1]=='NNP':
                plural_form=w[0]+'s'
                if plural_form in plural:
                    if w[0] in text_singular.keys():
                        text_singular[w[0]]+=1
                    if w[0] not in text_singular.keys():
                        text_singular.update({w[0]:1})
        return text_singular
    """
    help function for nouns_more_common_in_plural_form()
    returns bc_singular, a local dictionary
    Keys contain words that when appending an 's' to the word, the plural form is
    in text as well. Value is frequency of the singular words.
    """
    
    def tag_function(self):
        text_tag={}
        for w in self.tagged_text:
            if w[0] in text_tag.keys():
                text_tag[w[0]].append(w[1])
                
            if w[0] not in text_tag.keys():
                text_tag.update({w[0]:[w[1]]})
        return text_tag
    """
    returns text_tag, a local dictionary, with words in brown corpus as keys, tags (not distinct)
    as values. Helper function for proportion_ambiguous_word_tokens(self).
    """
    def get_tag(self):
        text_tagfreq={}
        for w in self.tagged_text:
            if w[1] in text_tagfreq.keys():
                text_tagfreq[w[1]]+=1
            if w[1] not in text_tagfreq.keys():
                text_tagfreq.update({w[1]:1})
        return text_tagfreq
    
    """
    returns text_tagfreq, a local dictionary, with distinct tag as keys, frequency of the tag as values.
    """
    
    def nouns_more_common_in_plural_form(self):
        result=[]
        text_plural=self.plural_words()
        text_singular=self.singular_words()
        for w, value in text_plural.items():
            w_sing=w[:-1]
            if w_sing not in text_singular.keys():
                result.append(w)
            else:
                if value>text_singular[w_sing]:
                    result.append(w)
        return result
    """ 
    Compare frequency of plural form of words and singular form, record the words
    that have higher frequency in plural form.
    If a word appears only as plural form in corpus, it is counted as a word
    that appears more in plural form.
    """
    
    def which_word_has_greatest_number_of_distinct_tags(self):
        maxval=0
        text_tag=self.tag_function()
        for w, value in text_tag.items():
            if w.isalpha():
                current = len(set(value))
                if current > maxval:
                    maxval=current
                    result=[]
                    result.append((w,list(set(value))))
                elif current == maxval:
                    result.append((w,list(set(value))))
            else:
                pass
        return result
    """
    call tag_function to return dic{word:tags}. Get number of distinct tags by using set. Record maximum
    number of distinct tags. Since word with most distinct tags is ']' without getting rid of punctuations, and
    ']' doesn't reveal anything. Get rid of puncpuations by using isalpha().
    """
    
    def tags_in_order_of_decreasing_frequency(self):
        x = self.get_tag()
        sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_x
    """
    call get_tag to return dic{tag: frequency}. First convert the dictionary to tuple and sort the tuple.
    """
    
    def tags_that_nouns_are_most_commonly_found_after(self):
        tagged=self.tagged_text
        tag_before_noun={}
        for w in range(len(tagged)-1):
            if tagged[w+1][1]in ['NN', 'NNS', 'NNP','NNPS']:
                if tagged[w][1] not in tag_before_noun.keys():
                    tag_before_noun.update({tagged[w][1]:1})
                else:
                    tag_before_noun[tagged[w][1]]+=1
        x=tag_before_noun
        sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_x
    """
    loop over self.tagged_text. If word+1 is a noun, record the word's tag in a dictionary.
    sort the dictionary by first converting it to tuple.
    
    """
    def proportion_ambiguous_word_types(self):
        data = nltk.ConditionalFreqDist((word.lower(), tag) 
                                         for (word, tag) in self.tagged_text)
        conditions = data.conditions()
        one_tag = [condition for condition in conditions if len(data[condition]) == 1]
        proportion_na = len(one_tag) / len(conditions)
        ambiguous = 1-proportion_na
        return ambiguous
    """
    use nltk.ConditionalFreqDist to record lower(words)' frequency conditioned on their tags.
    extract conditions by calling .conditions().
    record those with only one one tag.
    """
    


    def proportion_ambiguous_word_tokens(self):
        text_tag = self.tag_function()
        n=0
        for word, tag in text_tag.items():
                if len(set(tag))==1:
                    n+=len(tag)
        ambiguous = 1 - n/len(self.text)
        return ambiguous
    """ 
    call self.tag_function to return dic{word:tags}. If set(tags) is 1, then there
    is one tag for this word. Count total occurences of words with  one tag in brown corpus. 
    sum up all occurences of words with one tag.
    divided by len(bc.words) to calculate percentage of tokens with one tag. 1-that percentage
    gives percentage of ambiguous words.
    """
import sys
import argparse

from pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

""" import tagger model """

def run(sentence):
    tokens=sentence.split()
    return tagger.tag(tokens)
""" define a function so that an input sentence (string) can be run on the model. """
    
def tagger_test(browntype):
    return round(tagger.evaluate(brown.tagged_sents(categories = browntype)),3)
""" define a function so that a browncorpus category is read to run the model. """
    
    
    
def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagger-train", dest='taggertrain', action = "store_true")
    parser.add_argument("--tagger-run", dest='taggerrun')
    parser.add_argument("--tagger-test", dest='taggertest', type=str)

    args=parser.parse_args()
    
    if args.taggertrain:
        sys.stdout.write(str(tagger)+'\n')
    if args.taggerrun is not None:
        result=run(args.taggerrun)
        sys.stdout.write(str(result)+'\n')
    if args.taggertest is not None:
        evaluate = tagger_test(args.taggertest)
        sys.stdout.write(str(evaluate)+'\n')

if __name__=='__main__':
    Main()

""" Terminal interactive part."""      
        
