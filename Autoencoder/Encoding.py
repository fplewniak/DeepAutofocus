from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, image_size, channels=1, embedding_dim=10):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = (128, image_size, image_size)
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)

    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        # self.shape_before_flattening = tuple(x.shape[1:])
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x
