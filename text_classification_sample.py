#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:55:11 2017

@author: stepan
"""

from feature_extractors import bow_extractor, tfidf_extractor
from feature_extractors import averaged_word_vectorizer
from feature_extractors import tfidf_weighted_averaged_word_vectorizer
from sklearn.cross_validation import train_test_split
import nltk
import gensim
import pandas as pd

    
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels
    


import google_search as gs
import read_url
urls = gs.google_search('王将')
homepage_text = read_url.read_url(urls[0])

import re
p = re.compile(r"<[^>]*?>")
homepage_text = p.sub("", homepage_text) 
homepage_text = re.sub(r"\s|(&nbsp;)", "", homepage_text)

from normalization import normalize_corpus
homepage_text = homepage_text.strip()
homepage_text = homepage_text.replace("\xc2\xa0", "")
print(homepage_text)

import get_tokens as gt
data = gt.get_tokens(homepage_text)  
print(data)  

dataset = pd.DataFrame(columns =['target', 'data'])
df3 = pd.DataFrame([['中華料理', data]], columns =['target', 'data'])
dataset = dataset.append(df3)
dataset = dataset.append(df3)


corpus, labels = dataset.data, dataset.target


print('Sample document:', corpus[0])
print('Class label:',labels[0])

train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                        labels,
                                                                        test_data_proportion=0.3)
                                          
                                                     
from normalization import normalize_corpus


print(train_corpus)
print(test_corpus)
norm_train_corpus = normalize_corpus(train_corpus)
 #norm_test_corpus = normalize_corpus(test_corpus)  

''.strip()

print(norm_train_corpus)
# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)  
#bow_test_features = bow_vectorizer.transform(norm_test_corpus) 
print(bow_vectorizer)

# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)  
#tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)    


# tokenize documents
tokenized_train = [nltk.word_tokenize(text)
                   for text in norm_train_corpus]
tokenized_test = [nltk.word_tokenize(text)
                   for text in norm_test_corpus]  
# build word2vec model                   
model = gensim.models.Word2Vec(tokenized_train,
                               size=500,
                               window=100,
                               min_count=30,
                               sample=1e-3)                  
                   
# averaged word vector features
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=model,
                                                 num_features=500)   
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
                                                model=model,
                                                num_features=500)                                                 
                   


# tfidf weighted averaged word vector features
vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train, 
                                                                  tfidf_vectors=tfidf_train_features, 
                                                                  tfidf_vocabulary=vocab, 
                                                                  model=model, 
                                                                  num_features=500)
tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test, 
                                                                 tfidf_vectors=tfidf_test_features, 
                                                                 tfidf_vocabulary=vocab, 
                                                                 model=model, 
                                                                 num_features=500)


from sklearn import metrics
import numpy as np

def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        2))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
                        

def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)
    return predictions    

                        
               
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=100)

# Multinomial Naive Bayes with bag of words features
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)
                                           
# Multinomial Naive Bayes with tfidf features                                           
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with averaged word vector features
svm_avgwv_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=avg_wv_train_features,
                                           train_labels=train_labels,
                                           test_features=avg_wv_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with tfidf weighted averaged word vector features
svm_tfidfwv_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_wv_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_wv_test_features,
                                           test_labels=test_labels)

 

cm = metrics.confusion_matrix(test_labels, svm_tfidf_predictions)
pd.DataFrame(cm, index=range(0,20), columns=range(0,20))  

class_names = dataset.target_names
print(class_names[0], '->', class_names[15])
print(class_names[18], '->', class_names[16])
print(class_names[19], '->', class_names[15])



import re

num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 0 and predicted_label == 15:
        print('Actual Label:', class_names[label])
        print('Predicted Label:', class_names[predicted_label])
        print('Document:-')
        print(re.sub('\n', ' ', document))
        print
        num += 1
        if num == 4:
            break


num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 18 and predicted_label == 16:
        print('Actual Label:', class_names[label])
        print('Predicted Label:', class_names[predicted_label])
        print('Document:-')
        print(re.sub('\n', ' ', document))
        print
        num += 1
        if num == 4:
            break
