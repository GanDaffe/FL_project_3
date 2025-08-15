import torch
import torch.nn as nn
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 10)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(512, 47)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class CNN2(nn.Module):
    def __init__(self) -> None:
        super(CNN2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(5*5*64, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 model from torchvision
        self.resnet = models.resnet50(pretrained=False)
        
        # Adjust the first conv layer for CIFAR-100 (32x32 images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool to preserve spatial size
        
        # Replace the final fully connected layer for 100 classes
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    

class Moon_MLP(nn.Module): 

    def __init__(self) -> None:
        super(Moon_MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.l3 = nn.Linear(200, 10)

    def forward(self, x):
        h = self.features(x)
        y = self.l3(h)
        return h, h, y