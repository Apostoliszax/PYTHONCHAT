import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import BagOfWords, tokenize, stem
from model import NeuralNet


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
    X_train = []
    y_train = []

    for (patternSentence, tag) in xy:
        bag = BagOfWords(patternSentence, allWords)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    return np.array(X_train), np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def CreateDataloader(X_train, y_train, batchSize):
    dataset = ChatDataset(X_train, y_train)
    return DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f"Final loss: {loss.item():.4f}")
    return model


def SaveModel(model, inputSize, hiddenSize, outputSize, allWords, tags, filePath):
    data = {
        "model_state": model.state_dict(),
        "input_size": inputSize,
        "hidden_size": hiddenSize,
        "output_size": outputSize,
        "all_words": allWords,
        "tags": tags
    }
    torch.save(data, filePath)
    print(f"Training complete. Model saved to {filePath}")


def main():
    intents = LoadIntents('intents.json')
    ignoreWords = ['?', '.', '!', 'που', 'ποτε', 'ποιος', 'τι']

    allWords, tags, xy = PreprocessIntents(intents, ignoreWords)

    print(f"{len(xy)} patterns")
    print(f"{len(tags)} tags: {tags}")
    print(f"{len(allWords)} unique stemmed words")

    X_train, y_train = CreateTrainingData(xy, allWords, tags)

    epochs = 1000
    batchSize = 8
    learningRate = 0.001
    inputSize = len(X_train[0])
    hiddenSize = 8
    outputSize = len(tags)

    print(f"Input size: {inputSize}, Output size: {outputSize}")

    trainLoader = CreateDataloader(X_train, y_train, batchSize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    model = train(model, device, trainLoader, criterion, optimizer, epochs)

    SaveModel(model, inputSize, hiddenSize, outputSize, allWords, tags, "data.pth")


if __name__ == "__main__":
    main()
