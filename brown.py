
import nltk
from nltk.corpus import brown
import re
import operator

COMPILED_BROWN = 'brown.pickle'



class BrownCorpus(object):

    def __init__(self):
        self.words=list(word.lower() for word in brown.words())
        #self.words= brown.words()
        #self.text=nltk.Text(word.lower() for word in nltk.corpus.brown.words())
        self.tagged_words = brown.tagged_words()
        self.tagfreq={}
        self.raw=brown.raw()
        """
        words attribute is a list taking all words from brown.words().
        
        tagged_words is list taking all words with tags from brown.tagged_words().
        
        raw atrribute takes raw string from brown.raw().
        """
            
def plural_words(bc):
    bc_plural={}
    for w in bc.tagged_words:
        w_plur=w[0]
        if find_plural_pattern(w_plur):
            if w[0] in bc_plural.keys():
                if w[1]=='NNS' or w[1]=='NNPS':
                    bc_plural[w_plur]+=1
            if w[0] not in bc_plural.keys():
                if w[1]=='NNS' or w[1]=='NNPS':
                    bc_plural.update({w_plur:1})
    return bc_plural

    """
    help function for nouns_more_common_in_plural_form(bc)
    plural_word returns a dictionary as a local variable, key is plural words, value is frequency of 
    the plural words in brown corpus.
    """
        
def singular_words(bc):
    plural = plural_words(bc)
    bc_singular={}
    for w in bc.tagged_words:
        if w[1]=='NN' or w[1]=='NNP':
            plural_form=w[0]+'s'
            if plural_form in plural:
                if w[0] in bc_singular.keys():
                    bc_singular[w[0]]+=1
                if w[0] not in bc_singular.keys():
                    bc_singular.update({w[0]:1})
    return bc_singular
    """
    help function for nouns_more_common_in_plural_form(bc)
    returns bc_singular, a local dictionary
    Keys contain words that when appending an 's' to the word, the plural form is
    in brown corpus as well. Value is frequency of the singular words.
    """
        
def tag_function(bc):
    bc_tag={}
    for w in bc.tagged_words:
        if w[0] in bc_tag.keys():
            bc_tag[w[0]].append(w[1])
            
        if w[0] not in bc_tag.keys():
            bc_tag.update({w[0]:[w[1]]})
    return bc_tag

    """
    returns bc_tag, a local dictionary, with words in brown corpus as keys, tags (not distinct)
    as values.
    """
                
    
def get_tag(bc):
    bc_tagfreq={}
    for w in bc.tagged_words:
        if w[1] in bc_tagfreq.keys():
            bc_tagfreq[w[1]]+=1
        if w[1] not in bc_tagfreq.keys():
            bc_tagfreq.update({w[1]:1})
    return bc_tagfreq

    """
    returns bc_tagfreq, a local dictionary, with distinct tag as keys, frequency of the tag as values.
    """
            

def find_plural_pattern(word):
    plural=re.compile(r"\b(\w+)s\b")
    return bool(plural.match(word))

        
    """
    In this case, we simply assume any word ends with 's' is in plural form.
    """
    
def nouns_more_common_in_plural_form(bc):
    result=[]
    bc_plural=plural_words(bc)
    bc_singular=singular_words(bc)
    for w, value in bc_plural.items():
        w_sing=w[:-1]
        if w_sing not in bc_singular.keys():
            result.append(w)
        else:
            if value>bc_singular[w_sing]:
                result.append(w)
    return result
    """ 
    Compare frequency of plural form of words and singular form, record the words
    that have higher frequency in plural form.
    If a word appears only as plural form in corpus, it is counted as a word
    that appears more in plural form.
    It turns out there are 3150 words that appear more in plural form.
    """

def which_word_has_greatest_number_of_distinct_tags(bc):
    maxval=0
    bc_tag=tag_function(bc)
    for w, value in bc_tag.items():
        current = len(set(value))
        if current > maxval:
            maxval=current
            result=[]
            result.append((w,list(set(value))))
        elif current == maxval:
            result.append((w,list(set(value))))
    return result
    """
    call tag_function to return dic{word:tags}. Get number of distinct tags by using set. Record maximum
    number of distinct tags.
    """


def tags_in_order_of_decreasing_frequency(bc):
    x = get_tag(bc)
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x
    """
    call get_tag to return dic{tag: frequency}. First convert the dictionary to tuple and sort the tuple.
    """

def tags_that_nouns_are_most_commonly_found_after(bc):
    tagged=bc.tagged_words
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
    loop over bc.tagged_words. If word+1 is a noun, record the word's tag in a dictionary.
    sort the dictionary by first converting it to tuple.
    
    """


def proportion_ambiguous_word_types(bc):
    data = nltk.ConditionalFreqDist((word.lower(), tag)
                                    for (word, tag) in bc.tagged_words)
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

def proportion_ambiguous_word_tokens(bc):
    n=0
    bc_tag=tag_function(bc)
    for word, tag in bc_tag.items():
        if len(set(tag))==1:
            n+=len(tag)
    ambiguous = 1 - n/len(bc.words)
    return ambiguous
# =============================================================================
# =============================================================================


# =============================================================================
#     data = nltk.ConditionalFreqDist((word, tag)
#                                     for (word, tag) in bc.tagged_words)
#     ambiguous_words = [word.isalpha() for (word, tag) in data.items() if len(tag)>1]
#     ambiguous_tokens = [word for word in bc.words if word in ambiguous_words]
#     return len(ambiguous_tokens)/len(bc.words)
# =============================================================================

    
    
    """
    call tag_function to return dic{word:tags}. If set(tags) is 1, then there
    is one tag for this word. Count total occurences of words with  one tag in brown corpus. 
    sum up all occurences of words with one tag.
    divided by len(bc.words) to calculate percentage of tokens with one tag. 1-that percentage
    gives percentage of ambiguous words.
    """
        
        
    
    
    
    
    