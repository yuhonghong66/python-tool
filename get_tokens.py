#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:59:00 2017

@author: stepan

"""

def get_tokens(text):

  from janome.tokenizer import Tokenizer

  t = Tokenizer()
  tokens = t.tokenize(text.replace(',', ' '))
  words = []

  for token in tokens:
    if (token.surface is not None) and token.surface != "":
      words.append(token.surface)


  print(words)
  print(type(words))
  return words

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
 
wordsList = []
words = ' '.join(get_tokens(u'エスプレッソコーヒー'))
wordsList.append(words)
words = ' '.join(get_tokens(u'コーヒー'))
wordsList.append(words)
words = ' '.join(get_tokens(u'サンドイッチ'))
wordsList.append(words)
words = ' '.join(get_tokens(u'タマゴサンドイッチ'))
wordsList.append(words)
docs = np.array(wordsList)
print(docs)
 
#
# ベクトル化
#
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
print(vectorizer)
vecs = vectorizer.fit_transform(docs)
 
print(vecs.toarray())
#-----------------------------------------------------
# [[ 0.    0.    0.32  0.    0.    0.8   0.51  0.  ]
#  [ 0.    0.    0.41  0.65  0.    0.    0.65  0.  ]
#  [ 0.    0.    0.41  0.65  0.    0.    0.    0.65]
#  [ 0.69  0.    0.38  0.    0.    0.    0.    0.61]
#  [ 0.    0.    0.32  0.    0.8   0.    0.51  0.  ]
#  [ 0.49  0.57  0.27  0.43  0.    0.    0.    0.43]
#  [ 0.69  0.    0.38  0.    0.    0.    0.61  0.  ]
#  [ 0.    0.49  0.24  0.75  0.    0.    0.    0.37]]
#-----------------------------------------------------
 
 
#
# クラスタリング
#
clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vecs)
for doc, cls in zip(docs, clusters):
    print(cls, doc)
