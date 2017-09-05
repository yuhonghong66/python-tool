#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 21:57:39 2017

@author: stepan
"""

import nltk
import re
from html.parser import HTMLParser as html_parse


stopword_list = nltk.corpus.stopwords.words('english')


def keep_text_characters(text='Hello World'):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        print(token)
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
        filtered_text = ''.join(filtered_tokens)
        return filtered_text
    
def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    print(sentences)
    return sentences
    print(word_tokens)
    return word_tokens

def normalize_corpus(
        corpus,
        lemmatize=True,
        only_text_chars=False,
        tokeize=False):
    normalized_corpus = []
    for text in corpus:
        text = html_parse.unescape(text)
        text = expand_constractions(text, CONTRACTION_MAP)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove.special_characters(text)
        text = remove_stopwords(text)
        if only_text_chars:
            texy = keep_text_characters(text)
        if tokenize:
            text = tokenize_text(text)
        normalized_corpus.append(text)
    return normalized_corpus
    

def main():
    keep_text_characters()

if __name__ == "__main__":
    main()