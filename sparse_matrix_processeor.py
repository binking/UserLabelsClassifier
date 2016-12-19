# -*- coding: utf8 -*-
# refer:
#  1. Classification of text documents using sparse features
#        http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
#  2. 
import time
import jieba
import string
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
# from matplotlib import pyplot as plt
from sklearn.naive_bayes import (
	GaussianNB,
	MultinomialNB
)

def load_sparse_csr(filename):
    loader = np.load(filename)  # What is npz 
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)


def tokenize_text(text, cut_mode=True):
    """
    Split given text into tokens
    Here, comments and articles were separated by comma
    """
    tokens = []
    for sentence in text.split(','):
        for token in jieba.cut(sentence, cut_all=cut_mode):
            token = token.strip()
            if token not in string.punctuation:
                tokens.append(token)
    return tokens


def format_bow_csr_matrix(docs):
    index2word = {}
    print 'Format sparse matrix ...'
    data = []; indices = []; indptr = [0]
    idf_index2freq = {}
    for doc in docs:
        # print 'Cooking text'
        for token in doc:
            index = index2word.setdefault(token, len(index2word))  # how many words in vocabulary
            idf_index2freq[index] = idf_index2freq.get(index, 0) + 1
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))  # len(indices) means hwo many nnz in the col
    tf_csr_matrix = csr_matrix( (data, indices, indptr), dtype=int)
    idf_vector = np.array([freq for _, freq in sorted(idf_index2freq.items(), key=lambda x:x[0])])
    print idf_vector
    return tf_csr_matrix, idf_vector


def load_tokens_from_file(filename):
    docs = []
    print 'Loading data set ...'
    # <class 'pandas.core.frame.DataFrame'>
    # <class 'pandas.core.series.Series'> to <type 'numpy.ndarray'> : labels.values
    labels = []
    data_frame = pd.read_csv(filename, delimiter=',', nrows=1000)
    for i, doc in enumerate(data_frame['content']):
        if isinstance(doc, float):
            print i, '-th unkown doc: ', doc
            continue
        docs.append(tokenize_text(doc))
        labels.append(data_frame['class'][i])
    # import ipdb; ipdb.set_trace()  
    return docs, np.array(labels)


def main():
    load_time = time.time()
    # Preprocess text and generte csr matrix
    documents, labels = load_tokens_from_file('topic_classifier_simple_dataset.csv')
    term_freq, idf_vector = format_bow_csr_matrix(documents)
    td_idf_matrix = term_freq.dot(1/(1.0+idf_vector))
    size, dims = td_idf_matrix.shape
    train_set = td_idf_matrix[:int(0.8*size), :]  # slice sparse matrix: csc[:,indices], csr[indices,:]
    train_label = labels[:int(0.8*size)]
    train_time = time.time()
    print '\nPreprocess cost %d seconds' % (train_time - load_time)

    # Train model
    mnb_model = MultinomialNB()
    mnb_model.fit(train_set, train_label)
    prediction = mnb_model.predict(train_set)  # predict need dense matrix
    accuracy = np.sum(prediction==train_label)*1.0 / len(train_label)
    print "Accuracy is %f " % accuracy
    print '\nTrain data spent %d seconds' % (time.time() - train_time)

    # Test model
    test_set = td_idf_matrix[int(0.8*size):, :]
    test_labels = labels[int(0.8*size):]
    test_pred = mnb_model.predict(test_set)
    test_accuracy = np.sum(test_pred == test_labels)*1.0 / len(test_labels)
    print 'Test set accuracy: %f' % test_accuracy


if __name__=='__main__':
    main()
