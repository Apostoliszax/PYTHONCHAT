import torch.nn as nn

#TODO refactor
 
class NeuralNetwork(nn.Module):
    """
    A fully connected neural network with three hidden layers and dropout regularization.

    Args:
        inputSize (int): Number of input features.
        hiddenSize (int): Number of units in each hidden layer.
        numClasses (int): Number of output classes.
        dropout (float, optional): Dropout rate for regularization. Default is 0.3.
    """
    def __init__(self, inputSize, hiddenSize, numClasses, dropout=0.3):
        super(NeuralNetwork, self).__init__()
        self.fullyConnectedLayer1 = nn.Linear(inputSize, hiddenSize)
        self.activationFunction1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fullyConnectedLayer2 = nn.Linear(hiddenSize, hiddenSize)
        self.activationFunction2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fullyConnectedLayer3 = nn.Linear(hiddenSize, hiddenSize)
        self.activationFunction3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.out = nn.Linear(hiddenSize, numClasses)

    def forward(self, x):
        x = self.activationFunction1(self.fullyConnectedLayer1(x))
        x = self.dropout1(x)

        x = self.activationFunction2(self.fullyConnectedLayer2(x))
        x = self.dropout2(x)

        x = self.activationFunction3(self.fullyConnectedLayer3(x))
        x = self.dropout3(x)

        x = self.out(x)
        return x