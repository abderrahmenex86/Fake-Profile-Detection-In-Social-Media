import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from fake_profile_detector.configs.general import SAVE_DIR


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split, transform=None):
        self.dataset_split = dataset_split
        self.transform = transform
        self.label2idx = {"BOT": 0, "CYBORG": 1, "REAL": 2, "VERIFIED": 3}

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        image = self.dataset_split[idx]["image"].convert("RGB")
        label = self.dataset_split[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


class FakeProfileCNN(nn.Module):
    def __init__(self):
        super(FakeProfileCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 75 * 100, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 75 * 100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = load_dataset("drveronika/x_fake_profile_detection")
    train_dataset = CustomImageDataset(dataset["train"], transform=transform)
    val_dataset = CustomImageDataset(dataset["validation"], transform=transform)
    test_dataset = CustomImageDataset(dataset["test"], transform=transform)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeProfileCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")

    # save model
    torch.save(
        model.state_dict(), os.path.join(SAVE_DIR, "models", "fake_profile_cnn.pth")
    )


if __name__ == "__main__":
    main()
