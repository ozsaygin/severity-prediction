import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def tokenize_sentences(sentences):
    # takes list of summaries
    # returns a list of list which is tokenized version of each sentences
    tokenized_sentences = [[str(w) for w in word_tokenize(sent)] for sent in sentences]
    return tokenized_sentences

def remove_punctuation(summary_arr):
    arr = ["".join([ch for ch in text if ch not in string.punctuation]) for text in summary_arr]
    return arr

def remove_multipunc(summary_arr):
    sarr = []
    for sent in summary_arr:
        tmp1 = sent.split()
        tmp2 = []
        for word in tmp1:
            if 1 >= sum((word.count(chr) for chr in string.punctuation)):
                tmp2.append(word)
        sarr.append(" ".join(tmp2))
    return sarr
                

def remove_stopwords(summary_arr, tokenized):
    documents = []
    for bug_id in range(len(summary_arr)):
        docvec = []
        for w in tokenized[bug_id]:
            if w not in set(stopwords.words("english")):
                docvec.append(w)
        documents.append(docvec)
    return documents

def stem_words(summary_arr, documents):
    ps = PorterStemmer()
    stemmed_words = []
    for bug_id in range(len(summary_arr)):
        docvec = []
        for w in documents[bug_id]:
            if not any(char.isdigit() for char in w):
                docvec.append(ps.stem(w))
            elif w.startswith("0x"):
                docvec.append(ps.stem(w))
        stemmed_words.append(docvec)
    return stemmed_words

def convert_tolower(summary_arr):
    arr = []
    for text in summary_arr:
        tmp = []
        for ch in text:
            tmp.append(ch.lower())
        arr.append(''.join(tmp))
    return arr