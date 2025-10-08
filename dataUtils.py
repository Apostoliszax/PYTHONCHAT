import numpy as np
import json
from nltkUtils import BagOfWords, tokenize, stem

def LoadIntents(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        return json.load(file)

def PreprocessIntents(intents, ignoreWords):
    AllWords = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokenized = tokenize(pattern)
            AllWords.extend(tokenized)
            xy.append((tokenized, tag))
    AllWords = sorted(set(stem(w) for w in AllWords if w not in ignoreWords))
    tags = sorted(set(tags))
    return AllWords, tags, xy

def CreateTrainingData(xy, allWords, tags):
    X_data = []
    y_data = []
    for (patternSentence, tag) in xy:
        bag = BagOfWords(patternSentence, allWords)
        X_data.append(bag)
        label = tags.index(tag)
        y_data.append(label)
    return np.array(X_data), np.array(y_data)