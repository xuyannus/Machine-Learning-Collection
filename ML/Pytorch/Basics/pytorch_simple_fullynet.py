"""
Working code of a simple Fully Connected (FC) network training on MNIST dataset.
The code is intended to show how to create a FC network as well
as how to initialize loss, optimizer, etc. in a simple way to get
training to work with function that checks accuracy as well.

Video explanation: https://youtu.be/Jy4wM2X21u0
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-08 Initial coding

"""

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_mnist_data(batch_size=64):
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_hyper_parameters():
    return {
        "input_size": 784,
        "num_classes": 10,
        "learning_rate": 0.001,
        "num_epochs": 1,
        "batch_size": 64,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }


def train_model(loader, params):
    # hyper parameters
    input_size = params["input_size"]
    num_classes = params["num_classes"]
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]

    model = NN(input_size=input_size, num_classes=num_classes).to(params["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    print("Model training starts now...")
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(loader):
            # Get data to cuda if possible
            data = data.to(device=params["device"])
            targets = targets.to(device=params["device"])

            # Get to correct shape
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
    return model


def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()


def save_model(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    state = {"model_state": model.state_dict()}
    torch.save(state, filename)


if __name__ == "__main__":
    params = get_hyper_parameters()
    train_loader, test_loader = get_mnist_data(batch_size=params["batch_size"])
    model = train_model(train_loader, params)
    check_accuracy(train_loader, model, device=params["device"])
    check_accuracy(test_loader, model, device=params["device"])
    save_model(model)
