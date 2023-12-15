import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3780, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(trainloader, epochs, lr):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    accuracys = []

    for epoch in range(epochs):  # loop over the dataset for the specified number of epochs
        running_loss = 0.0
        total = 0
        correct = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # calculate and print epoch loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        accuracys.append(epoch_accuracy)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%')

    print('Finished Training')
    return net, accuracys


def test(test_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy of the network on the test data: %.4f %%' % (
            100 * accuracy))
    return accuracy


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
