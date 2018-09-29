#!/anaconda3/envs/bugzilla-env/bin/python

# !! change python environment !!

import data_helpers
import gensim
import getopt
import numpy as np
import pandas as pd
import os
import pathlib
import pickle
import sklearn
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *
from sklearn.metrics import *


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        This method sums all wordvecs of all words in a sentences
        and divides the resulting vector by the len of word count in the sentence
        """
        return np.array([np.sum([self.word2vec[w] for w in words if w in self.word2vec] or
                                [np.zeros(100)], axis=0) / len(words) for words in X])


def train_and_evaluate(inputfile):
    print('Importing bug reports...')
    df = pd.read_csv(inputfile, error_bad_lines=False, quotechar="'", encoding='utf-8')
    print("Total number of sentences: ", df.size)  # number of verified summaries by developers

    class_names = ["non-severe", "normal", " severe"]

    print('Preprocessing summaries...(This may take several minutes, please wait!)   ')
    summaries = [df.values[id][0] for id in range(len(df.values))]
    severities = [df.values[id][1] for id in range(len(df.values))]

    summary_arr = data_helpers.convert_tolower(summaries)
    summary_arr = data_helpers.remove_punctuation(summary_arr)
    sentences = data_helpers.tokenize_sentences(summary_arr)
    # documents = remove_stopwords(summary_arr, tokenized)
    # stemmed = stem_words(summary_arr, documents)

    print('Calculating word vectors...')
    model = gensim.models.Word2Vec(sentences, size=100, workers=-1, iter=1000)
    words = list(model.wv.vocab)
    model.wv.save_word2vec_format('w2v_model.bin')

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    mev = MeanEmbeddingVectorizer(w2v)
    M = mev.transform(sentences)
    y = np.array(severities)

    print('Training model... This process may take severak minutes...')
    nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='lbfgs',
                                              random_state=1,
                                              verbose=True, early_stopping=False, max_iter=1000)

    # validation (comment out this block if you don't want to validate)
    print('Validating the neural network model...')
    skf = StratifiedKFold(n_splits=5)
    total_accuracy = 0
    for train_index, test_index in skf.split(M, y):
        x_train, x_test = M[train_index], M[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn.fit(x_train, y_train)
        y_predict = nn.predict(x_test)
        total_accuracy += accuracy_score(y_test, y_predict)

    # train
    nn.fit(M, y)
    with open('nn_model.bin', 'wb') as file:
        pickle.dump(nn, file)

    with open('w2v_model.bin', 'wb') as file:
        pickle.dump(model, file)

    print('Models are written to disk...')

    # comment out this line if you comment out validation block
    print("Accuracy: %2.1f %%" % (total_accuracy / 5 * 100))


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        print ('train_and_validate.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('train_and_validate.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    if inputfile == '':
        inputfile = 'summaryList.csv'

    train_and_evaluate(inputfile)


if __name__ == '__main__':
    main(sys.argv[1:])

