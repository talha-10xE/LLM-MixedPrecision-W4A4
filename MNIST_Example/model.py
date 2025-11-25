import torch.nn as nn

class SimpleDenseNet(nn.Module):
    """
    A simple fully connected (dense) neural network for MNIST classification.
    Layer structure: 784 -> 300 -> 70 -> 10.
    """
    def __init__(self):
        super(SimpleDenseNet, self).__init__()
        
        # 28*28 = 784 input features (flattened image)
        self.fc1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(300, 70)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(70, 10) # 10 output classes (digits 0-9)
        
    def forward(self, x):
        # x shape: (batch_size, 784)
        
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Output layer (no activation, CrossEntropyLoss handles logits)
        x = self.fc3(x) 
        
        return x