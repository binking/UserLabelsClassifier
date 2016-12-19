# -*- coding: utf8 -*-
# refer:
#  1. Classification of text documents using sparse features
#        http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
#  2. 
import time
import jieba
import string
import traceback
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix, lil_matrix
# from matplotlib import pyplot as plt
from sklearn.naive_bayes import (
	GaussianNB,
	MultinomialNB
)
from sklearn.linear_model import LogisticRegression

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
    return tf_csr_matrix, idf_vector


def load_tokens_from_file(filename):
    docs = []
    print 'Loading data set ...'
    # <class 'pandas.core.frame.DataFrame'>
    # <class 'pandas.core.series.Series'> to <type 'numpy.ndarray'> : labels.values
    labels = []
    data_frame = pd.read_csv(filename, delimiter=',') # , nrows=5000)  # sample the whole: df.sample(frac=1)
    random_frame = data_frame.reindex(np.random.permutation(data_frame.index))
    for i, doc in enumerate(random_frame['content']):
        if isinstance(doc, float) and random_frame['class'][i] == '其他'.decode('utf8'):
            print i, '-th doc ignored'
            continue
        docs.append(tokenize_text(doc))
        labels.append(random_frame['class'][i])
        if i > 999:
            break
    # import ipdb; ipdb.set_trace()  
    return docs, np.array(labels)


def give_me_classifier(X_train, y_train, X_test, y_test, clf):
    print '_' * 80
    print "Training with %s : " % clf.__name__
    t0 = time()
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    train_acc = np.sum(y_train==train_pred)*1.0 / len(train_pred)
    print 'Train Accuracy is %2.3f' % (train_acc * 100)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time 

    t0 = time()
    test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test==test_pred)*1.0 / len(test_pred)
    print "Test Accuracy is %2.3f" % (test_acc * 100)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    return clf_descr, train_acc, test_acc, train_time, test_time


def main():
    load_time = time.time()
    # Preprocess text and generte csr matrix
    documents, labels = load_tokens_from_file('topic_classifier_simple_dataset.csv')
    try:
        term_freq, df_vector = format_bow_csr_matrix(documents)
    except MemoryError as e:
        traceback.print_exc()
        return -1
    size, dims = term_freq.shape
    print 'Scale: ', size, dims
    idf_vector = 1.0/(1+df_vector)
    idf_matrix = lil_matrix((dims, dims))
    idf_matrix.setdiag(idf_vector)
    td_idf_matrix = term_freq * idf_matrix
    # split dataset into 2 parts with rate of 8:2
    split_line = int(0.8*size)
    train_set = td_idf_matrix[:split_line, :]  # slice sparse matrix: csc[:,indices], csr[indices,:]
    train_label = labels[:split_line]
    test_set = td_idf_matrix[split_line:, :]
    test_labels = labels[split_line:]
    train_time = time.time()
    print '\nPreprocess cost %d seconds' % (train_time - load_time)

    results = []
    for clf, name in (
        (MultinomialNB(alpha=0.2), "Multinomial Navive Bayes(alpha=0.2)"),
        (MultinomialNB(alpha=0.4), "Multinomial Navive Bayes(alpha=0.4)"),
        (MultinomialNB(alpha=0.6), "Multinomial Navive Bayes(alpha=0.6)"),
        (MultinomialNB(alpha=0.8), "Multinomial Navive Bayes(alpha=0.8)"),
        (MultinomialNB(), "Multinomial Navive Bayes(alpha=1,default)"),
        (LogisticRegression(), "Logistic regression(L2 penalty)"),
        (LogisticRegression(penalty='l1'), "Logistic regression(L1 penalty)"),
        (LogisticRegression(C=1., solver='lbfgs'), "Logistic regression with no calibration as baseline")
        # (KNeighborsClassifier(n_neighbors=10), "kNN"),
        # (RandomForestClassifier(n_estimators=100), "Random forest")
        )[:2]:
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))
    print "=" * 80
    for res in results:
        print 'Classifier: %s, its train accuracy = %2.3f, test accuracy = %2.3f' % res[0], res[1], res[2]
        print 'And it cost %f for training and cost %f for testing' % (res[3], res[4])
    

if __name__=='__main__':
    main()
