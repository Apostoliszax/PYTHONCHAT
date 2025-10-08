import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
    X_data = []
    y_data = []
    for (patternSentence, tag) in xy:
        bag = BagOfWords(patternSentence, allWords)
        X_data.append(bag)
        label = tags.index(tag)
        y_data.append(label)
    return np.array(X_data), np.array(y_data)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y
    def __getitem__(self, index):
        return torch.tensor(self.x_data[index], dtype=torch.float32), torch.tensor(self.y_data[index], dtype=torch.long)
    def __len__(self):
        return len(self.x_data)

def CreateDataloader(X, y, batchSize):
    dataset = ChatDataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)

def TrainValidateSplit(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def evaluate(model, device, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for words, labels in data_loader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for words, labels in train_loader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            val_loss, val_acc = evaluate(model, device, val_loader, criterion)
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}")
    final_val_loss, final_val_acc = evaluate(model, device, val_loader, criterion)
    print(f"Final Val Loss: {final_val_loss:.4f} | Final Val Acc: {final_val_acc:.2f}")
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
    X, y = CreateTrainingData(xy, allWords, tags)
    X_train, X_val, y_train, y_val = TrainValidateSplit(X, y, test_size=0.2)
    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")
    epochs = 1000
    batchSize = 8
    learningRate = 0.001
    inputSize = len(X_train[0])
    hiddenSize = 64
    outputSize = len(tags)
    print(f"Input size: {inputSize}, Output size: {outputSize}")
    trainLoader = CreateDataloader(X_train, y_train, batchSize)
    valLoader = CreateDataloader(X_val, y_val, batchSize)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hiddenSize = 64
    model = NeuralNet(inputSize, hiddenSize, outputSize, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    model = train(model, device, trainLoader, valLoader, criterion, optimizer, epochs)
    SaveModel(model, inputSize, hiddenSize, outputSize, allWords, tags, "data.pth")

if __name__ == "__main__":
    main()
