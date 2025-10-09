from sklearn.model_selection import train_test_split
import torch

def TrainValidateSplit(X, y, testSize=0.2):
    return train_test_split(X, y, test_size=testSize, random_state=42)

def evaluate(model, device, dataLoader, criterion):
    model.eval()
    totalLoss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for words, labels in dataLoader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            totalLoss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = totalLoss / len(dataLoader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train(model, device, trainLoader, valueLoader, criterion, optimizer, numEpochs, evaluateFn):
    for epoch in range(numEpochs):
        model.train()
        for words, labels in trainLoader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            valLoss, valAcc = evaluateFn(model, device, valueLoader, criterion)
            print(f"Epoch [{epoch + 1}/{numEpochs}] | Train Loss: {loss.item():.4f} | Val Loss: {valLoss:.4f} | Val Acc: {valAcc:.2f}")
    finalValLoss, finalValAcc = evaluateFn(model, device, valueLoader, criterion)
    print(f"Final Val Loss: {finalValLoss:.4f} | Final Val Acc: {finalValAcc:.2f}")
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