
import dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import MNIST
from model import LeNet5, CustomMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, Dropout
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

    train_dataset = dataset.MNIST('../data/train')
    test_dataset = dataset.MNIST('../data/test')
 
    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    models = {'LeNet5': LeNet5(), 'CustomMLP': CustomMLP()}
    for model_name, model in models.items():
        model.to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = CrossEntropyLoss()

        # Lists to store metrics for plotting
        training_losses, training_accuracies = [], []
        test_losses, test_accuracies = [], []

        epochs = 30
        for epoch in range(epochs):
            trn_loss, trn_acc = train(model, trn_loader, device, criterion, optimizer)
            tst_loss, tst_acc = test(model, tst_loader, device, criterion)
            
            # Store metrics
            training_losses.append(trn_loss)
            training_accuracies.append(trn_acc)
            test_losses.append(tst_loss)
            test_accuracies.append(tst_acc)

            print(f'{model_name} - Epoch {epoch+1}/{epochs} - Training Loss: {trn_loss:.4f}, Accuracy: {trn_acc:.4f}')
            print(f'{model_name} - Epoch {epoch+1}/{epochs} - Test Loss: {tst_loss:.4f}, Accuracy: {tst_acc:.4f}')
        
        # Plotting the metrics
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title(f'Loss Over Time for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(training_accuracies, label='Training Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title(f'Accuracy Over Time for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.suptitle(f'Training and Testing Metrics for {model_name}')
        plt.show()

if __name__ == '__main__':
    main()
