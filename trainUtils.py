from sklearn.model_selection import train_test_split
import torch

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

def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, evaluate_fn):
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
            val_loss, val_acc = evaluate_fn(model, device, val_loader, criterion)
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}")
    final_val_loss, final_val_acc = evaluate_fn(model, device, val_loader, criterion)
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