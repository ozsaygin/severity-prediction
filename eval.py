#!/anaconda3/envs/bugzilla-env/bin/python

# !! change python environment !!

import data_helpers
import numpy as np
import pickle

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

def eval(summary):

    with open('nn_model.bin', 'rb') as file:
        nn = pickle.load(file)

    with open('w2v_model.bin', 'rb') as file:
       w2v_model =  pickle.load(file) 

    summary_arr = data_helpers.convert_tolower([summary])
    summary_arr = data_helpers.remove_punctuation(summary_arr)
    sentences = data_helpers.tokenize_sentences(summary_arr)

    w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
    mev = MeanEmbeddingVectorizer(w2v)
    M = mev.transform(sentences)


    prediction = nn.predict(M)
    print('Severity: %s' % prediction[0])

if __name__ == '__main__':
    summary = ''
    while summary != 'quit':
        summary = input("Please enter your bug summary, or enter 'quit': ")
        if summary == '' :
            continue
        elif summary != 'quit':
            eval(summary)

