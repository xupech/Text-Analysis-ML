#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:14:11 2019

@author: xupech
"""

import sys
import unittest
import warnings

from main_a4 import Text

def ignore_warnings(test_func):
    """Catching warnings via a decorator."""
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test

class ExploreTextTests(unittest.TestCase):

    @classmethod
    @ignore_warnings
    def setUpClass(cls):
        cls.grail = Text('data/grail.txt')
        cls.nouns_more_common_in_plural=cls.grail.nouns_more_common_in_plural_form()
        cls.most_tags = cls.grail.which_word_has_greatest_number_of_distinct_tags()
        cls.frequent_tags = cls.grail.tags_in_order_of_decreasing_frequency()
        cls.tags_after_nouns = cls.grail.tags_that_nouns_are_most_commonly_found_after()
        cls.ambiguous_types = cls.grail.proportion_ambiguous_word_types()
        cls.ambiguous_tokens = cls.grail.proportion_ambiguous_word_tokens()

    def test_nouns_more_common_in_plural(self):
        """There are about 95 of them."""
        self.assertTrue(90 < len(self.nouns_more_common_in_plural) < 100)
        
    def test_most_tags(self):
        """The word with the most tags is 'ARTHUR'."""
        self.assertEqual(self.most_tags[0][0], 'ARTHUR')
        
    def test_frequent_tags1(self):
        """The most frequent Brown tag is NN and it occurs 2284 times."""
        self.assertEqual(self.frequent_tags[0][0], 'NN')
        self.assertTrue(2000 < self.frequent_tags[0][1], 2400)

    def test_frequent_tags2(self):
        """Get the 20 most frequent tags and make sure the overlap is greater than 18."""
        most_frequent_tags = {'NN', 'NNP', '.', ':', 'PRP', 'DT', 'JJ', ',', 'IN',
                               'RB', 'VB', 'VBP', 'VBZ', 'CC', 'NNS', 'UH', 'VBD', 'CD',
                              'PRP$',  'MD'}
        most_frequent_tags_found = set(t[0] for t in self.frequent_tags[:20])
        self.assertTrue(len(most_frequent_tags & most_frequent_tags_found) > 18)
        
    def test_noun_tags1(self):
        """Overlap of found set and example set is at least 8."""
        tags = [('.', 1198), ('NNP', 588), (':', 576), ('DT', 527), ('JJ', 443),
                ('NN', 426), ('IN', 167), ('PRP$', 125), (',', 115), ('CC', 49)]
        self.assertTrue(len(set([t[0] for t in tags])
                            & set([t[0] for t in self.tags_after_nouns])) > 8)
    
    def test_noun_tags2(self):
        """Most frequent tag before a noun occurs at least 1000 times."""
        self.assertTrue(self.tags_after_nouns[0][1] > 1000)

    def test_ambiguous_types(self):
        self.assertTrue(0.18 < self.ambiguous_types < 0.22)

    def test_ambiguous_tokens(self):
        self.assertTrue(0.40 < self.ambiguous_tokens < 0.45)


if __name__ == '__main__':

    unittest.main()