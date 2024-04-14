
import dataset
import numpy as np
import torch
from dataset import MNIST
from model import LeNet5, CustomMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    total_loss, total_correct, total_items = 0, 0, 0
    for inputs, labels in trn_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_items += labels.size(0)
    trn_loss = total_loss / total_items
    acc = total_correct / total_items

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    total_loss, total_correct, total_items = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tst_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_items += labels.size(0)
    tst_loss = total_loss / total_items
    acc = total_correct / total_items

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create Dataset objects
    train_dataset = dataset.MNIST('./data/train')
    test_dataset = dataset.MNIST('./data/test')
    # Create DataLoader objects
    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Create model, optimizer, and cost function
    model = LeNet5().to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()
    # Train and test the model
    trn_loss, acc = train(model, trn_loader, device, criterion, optimizer)
    print(f'Training Loss: {trn_loss:.4f}, Accuracy: {acc:.4f}')
    tst_loss, acc = test(model, tst_loader, device, criterion)
    print(f'Test Loss: {tst_loss:.4f}, Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
