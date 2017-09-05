#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 21:26:52 2017

@author: stepan
"""

import gensim
import nltk
nltk.download()

CORPUS = [
        'ths sky is blue',
        'sky is blue and sky is beautiful',
        'the beautiful sky is so blue',
        'i love blue cheese']
new_doc =['loving this blue sky today']

TOKENIZED_CORPUS = [nltk.word_tokenize(sentence) for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence) for sentence in new_doc]

model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=10, window=10, min_count=2, sample=1e-3)

print(model.)