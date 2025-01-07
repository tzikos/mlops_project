import torch
from torch import nn


class dreamer(nn.Module):
    """Implementation of my model for the MNIST dataset.
    
    The model architecture is as follows:
    - Convolutional layer with 32 filters of size 3x3
    - ReLU activation
    - Max pooling with kernel size 2x2
    - Convolutional layer with 64 filters of size 3x3
    - ReLU activation
    - Max pooling with kernel size 2x2
    - Convolutional layer with 128 filters of size 3x3
    - ReLU activation
    - Max pooling with kernel size 2x2
    - Flatten layer
    - Dropout with probability 0.5
    - Fully connected layer with 10 output units

    Functions:
    - forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters:
        - x: Input tensor of shape (batch_size, 1, 28, 28)
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = dreamer()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

