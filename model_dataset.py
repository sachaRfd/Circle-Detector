import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from functions import *

class CircleDataset(Dataset):
    def __init__(self, noise_level, img_size, min_radius=None, max_radius=None):
        self.generator = generate_examples(noise_level=noise_level, img_size=img_size, min_radius=min_radius, max_radius=max_radius)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor for pytorch implementation
            transforms.Normalize(0.5, 0.5),  # Preprocessing pixel
        ])
        self.length = 1000  # size of the dataset - here we chose 1_000 for small dataset

    def __getitem__(self, idx):
        img, params = next(self.generator)
        img_tensor = self.transform(img)
        params_tensor = torch.tensor([params.row, params.col, params.radius])
        return img_tensor, params_tensor

    def __len__(self):
        return self.length  # always return length


class circle_model(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(circle_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),  # Gereralisation technique
            nn.Mish(),  # Mish function shows self-regularising properties
            nn.MaxPool2d(2),  # downsample and reduce spatial dimension
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 + 128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256 + 128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(dropout_rate),  # Another generalisation technique
        )
        self.fc5 = nn.Sequential(
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = x.float()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.view(-1, 128 * 6 * 6)  # change into linear setup
        x5 = self.fc1(x4)
        x6 = torch.cat((x4, x5), dim=1)  # concatenate skip-connections
        x7 = self.fc2(x6)
        x8 = self.fc3(x7)
        x9 = torch.cat((x5, x8), dim=1)
        x10 = self.fc4(x9)
        x11 = self.fc5(x10)
        return x11

