import torch
from dataUtils import LoadIntents, PreprocessIntents, CreateTrainingData
from dataset import CreateDataloader
from trainUtils import TrainValidateSplit, evaluate, train, SaveModel
from model import NeuralNetwork

def main():
    intents = LoadIntents('intents.json')
    ignoreWords = ['?', '.', '!', 'που', 'ποτε', 'ποιος', 'τι']
    allWords, tags, xy = PreprocessIntents(intents, ignoreWords)
    print(f"{len(xy)} patterns")
    print(f"{len(tags)} tags: {tags}")
    print(f"{len(allWords)} unique stemmed words")
    X, y = CreateTrainingData(xy, allWords, tags)
    X_train, X_val, y_train, y_val = TrainValidateSplit(X, y, testSize=0.2)
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
    model = NeuralNetwork(inputSize, hiddenSize, outputSize, dropout=0.3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    model = train(model, device, trainLoader, valLoader, criterion, optimizer, epochs, evaluate)
    SaveModel(model, inputSize, hiddenSize, outputSize, allWords, tags, "data.pth")

if __name__ == "__main__":
    main()