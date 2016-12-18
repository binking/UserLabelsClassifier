import csv
import json
import jieba
import sframe
import string
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

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

def gen_bag_of_words(texts):
    """
    Given list of strings, generate bags of words
    """
    bag_of_words = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        bag_of_words.update(tokens)
    return bag_of_words
        
def load_dataset(filename):
    """
    Load given file with SFrame
    """
    data_frame = sframe.SFrame()
    train_data = sframe.SFrame()
    dataset = data_frame.read_csv(filename, delimiter='|', header=False, nrows=100)
    train_data['topics'] = dataset['X1']
    train_data['content'] = dataset['X2']
    return train_data, dataset['X3']

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix( (loader['data'], loader['indices'], loader['indptr']), loader['shape'])

def save_word_count(filename, bag_of_words):
    pass

def main():
	filename = './user_labels_simple_dataset.csv'  # 3428 lines
	train_data, labels = load_dataset(filename)
	bag_of_voca = gen_bag_of_words(train_data['content']).items()  # later can't change this, cause it's in order
	index_word_dict = dict((k, t[0]) for k, t in enumerate(bag_of_voca))

	# Gen sparse matrix
	index_freq_matrix = []
	inverse_docu_freq_vector = np.zeros((len(bag_of_voca), )) 
	for i in range(len(train_data['content'])):
	    print i,   # remember: when you had a long loop, display its process
	    word_freq_dict = gen_bag_of_words([train_data['content'][i]])  # return Counter() object
	    for j in range(len(bag_of_voca)):
	        word, _ = bag_of_voca[j]
	        if word_freq_dict[word]:  # if the word exists in the document
	            index_freq_matrix.append([i, j, word_freq_dict[word]])
	            inverse_docu_freq_vector[j] += 1
	sparse_matrix = csr_matrix(index_freq_matrix)


"""
# Back of
# transfer bag of words model to numerical matrix
term_freq_matrix = np.zeros((len(labels), len(bag_of_voca)))
inverse_docu_freq_vector = np.zeros((len(bag_of_voca), )) 
for i, line in enumerate(train_data['content']):
    print i,   # remember: when you had a long loop, display its process
    word_freq_dict = gen_bag_of_words([line])  # return Counter() object
    for j in range(len(bag_of_voca)):
        word, _ = bag_of_voca[j]
        if word_freq_dict[word]:  # if the word exists in the document
            term_freq_matrix[i, j] = word_freq_dict[word]
            inverse_docu_freq_vector[j] += 1
# save matrix into csv format
tf_dataframe = sframe.SFrame()
idf_dataframe = sframe.SFrame()
for j in range(len(bag_of_voca)):
    print j,
    word, _ = bag_of_voca[j]
    tf_dataframe[word.encode('utf8')] = term_freq_matrix[:, j]
idf_dataframe['df'] = inverse_docu_freq_vector
tf_dataframe.save('data/term_freq_matrix.csv', format='csv')
idf_dataframe.save('data/docu_freq_matrix.csv', format='csv')
# td_idf_matrix = np.dot(term_freq_matrix, inverse_docu_freq_vector)
"""