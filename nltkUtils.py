import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def BagOfWords(tokenizedSentence, words):
    sentenceWords = [stem(word) for word in tokenizedSentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for index, word in enumerate(words):
        if word in sentenceWords: 
            bag[index] = 1
    return bag