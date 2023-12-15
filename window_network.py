import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(3780, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_overlapping(trainloader, epochs, lr):
    epoch_losses = []
    net = Net1()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.view(-1, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update running loss
            running_loss += loss.item()

        # print average loss per epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1} - Loss: {epoch_loss:.5f}')

    print('Finished Training')
    return net, epoch_losses


def test_overlapping(test_loader, net):
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.view(-1, 1)
            outputs = net(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Calculate average loss over all the test data
    avg_loss = total_loss / len(test_loader)
    print(f'Average loss on the test data: {avg_loss:.5f}')

    return avg_loss


def train(trainloader, epochs, lr):
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update running loss
            running_loss += loss.item()

        # print average loss per epoch
        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1} - Loss: {epoch_loss:.5f}')

    print('Finished Training')
    return net


def test(test_loader, net):
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            print(outputs, labels)
            # Calculate the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Calculate average loss over all the test data
    avg_loss = total_loss / len(test_loader)
    print(f'Average loss on the test data: {avg_loss:.5f}')

    return avg_loss


def grid_search(train_loader, test_loader, epochs_list, lr_list):
    best_accuracy = 0
    best_params = {}

    for epochs in epochs_list:
        for lr in lr_list:
            print(f"Training with epochs={epochs}, lr={lr}")

            # Train the model with the specified parameters
            trained_net = train(train_loader, epochs, lr)

            # Test the model
            accuracy = test(test_loader, trained_net)

            # Update the best parameters if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'epochs': epochs, 'lr': lr}

    return best_accuracy, best_params


def plot_metrics(epoch_accuracies):  # , epoch_accuracies
    plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.plot(epoch_losses, label='Loss')
    # plt.title('Epoch vs Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # Uncomment and modify this part if you're calculating accuracy
    # plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies, label='Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
