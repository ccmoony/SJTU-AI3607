from data import CIFAR10
from model import DeepPermNet
from jittor import nn
import numpy as np
import jittor as jt
from jittor.optim import Adam
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, epoch, losses_list, accu_list, patch_num = 4):
    model.train()
    losses = 0
    puzzle_acc = 0
    fragment_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        total_num += 1
        batch_size = inputs.shape[0]
        pred = model(inputs)
        pred = pred.view(-1, patch_num)
        loss = nn.cross_entropy_loss(pred, targets.flatten())
        pred = np.argmax(pred.numpy(), axis=1)
        pred = pred.reshape(-1,patch_num)
        pred = np.sum(targets.numpy()==pred,axis=1)
        fragment_acc += np.sum(pred)/(batch_size*patch_num)
        pred = (pred == patch_num)
        puzzle_acc += np.sum(pred)/batch_size
        optimizer.step(loss)
        losses += loss.numpy()[0]
    puzzle_acc = puzzle_acc/total_num
    fragment_acc = fragment_acc/total_num
    losses_list.append(losses)  
    accu_list.append(fragment_acc)  
    print('Train Acc =', puzzle_acc, 'Fragment Acc =', fragment_acc)
    print(f"Train Epoch {epoch} Loss: {losses}")
            
def test(model, val_loader, epoch, losses_list, accu_list, patch_num = 4):
    model.eval()
    losses = 0
    puzzle_acc = 0
    fragment_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        total_num += 1
        batch_size = inputs.shape[0]
        pred = model(inputs)
        pred = pred.view(-1, patch_num)
        loss = nn.cross_entropy_loss(pred, targets)
        losses += loss.numpy()[0]
        pred = np.argmax(pred.numpy(), axis=1)
        pred = pred.reshape(-1,patch_num)
        pred = np.sum(targets.numpy()==pred,axis=1)
        fragment_acc += np.sum(pred)/(batch_size*patch_num)
        pred = (pred == patch_num)
        puzzle_acc += np.sum(pred)/batch_size
    puzzle_acc = puzzle_acc/total_num
    fragment_acc = fragment_acc/total_num
    losses_list.append(losses)  
    accu_list.append(fragment_acc) 	
    print('Test Acc =', puzzle_acc, 'Fragment Acc =', fragment_acc)
    print('Loss = ', losses)            

def plot_loss(Train_losses, Test_losses):
    Epochs = [i+1 for i in range(len(Train_losses))]
    plt.plot(Epochs, Train_losses, label="Train")
    plt.plot(Epochs, Test_losses, label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

def plot_acc(Train_accu, Test_accu):
    Epochs = [i+1 for i in range(len(Train_accu))]
    plt.plot(Epochs, Train_accu, label="Train")
    plt.plot(Epochs, Test_accu, label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

def main(batch_size = 1024,learning_rate = 1e-2, momentum = 0.9, weight_decay = 1e-4, num_epochs = 200, patch_size=(16,16), patch_num=4):
    Train_losses = []
    Train_accu = []
    Test_accu = []
    Test_losses = []
    model = DeepPermNet(patch_size=patch_size)
    train_loader = CIFAR10(phase='Train',  batch_size=batch_size, patch_size=patch_size)
    val_loader = CIFAR10(phase='Test',  batch_size=batch_size, patch_size=patch_size)
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch, Train_losses, Train_accu ,patch_num)
        test(model, val_loader, epoch, Test_losses, Test_accu, patch_num)
    plot_loss(Train_losses, Test_losses)
    plot_acc(Train_accu, Test_accu)


if __name__ == '__main__':
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    main(batch_size=500,learning_rate=1e-2,num_epochs=30,patch_size=(8,16),patch_num=8)