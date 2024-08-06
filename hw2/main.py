from data import CIFAR10
from model import CNN,RNN,Resnet
from jittor import nn
import numpy as np
import jittor as jt
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, epoch, losses_list, accu_list):
    model.train()
    total_acc = 0
    total_num = 0
    losses = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        loss = nn.cross_entropy_loss(outputs, targets)
        total_acc += acc
        total_num += batch_size
        optimizer.step (loss)
        losses += loss.numpy()[0]
    losses_list.append(losses)  
    print(f'Train Acc: {total_acc/total_num}')
    accu_list.append(total_acc/total_num)  
    print(f"Train Epoch {epoch} Loss: {losses}")
            
def val(model, val_loader, epoch, losses_list, accu_list):
    model.eval()
    total_acc = 0
    total_num = 0
    losses = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        losses += loss.numpy()[0]
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
    losses_list.append(losses)  
    accu_list.append(total_acc/total_num) 	
    print('Test Acc =', total_acc / total_num)
    print('Loss = ', losses)            

def plot_loss(Train_losses, Test_losses, reduced=False):
    Epochs = [i+1 for i in range(len(Train_losses))]
    plt.plot(Epochs, Train_losses, label="Train")
    plt.plot(Epochs, Test_losses, label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    if reduced:
        plt.title('Loss on reduced dataset')
    else:
        plt.title('Loss')
    plt.show()

def plot_acc(Train_accu, Test_accu,reduced=False):
    Epochs = [i+1 for i in range(len(Train_accu))]
    plt.plot(Epochs, Train_accu, label="Train")
    plt.plot(Epochs, Test_accu, label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    if reduced:
        plt.title('Accuracy on reduced dataset')
    else:
        plt.title('Accuracy')
    plt.show()

def main(batch_size = 1024,learning_rate = 1e-2, momentum = 0.9, weight_decay = 1e-4, epochs = 200, model_name='CNN',reduced=False,enhanced=True):
    '''
    Args:
    reduced: If use the reduced dataset
    enhanced: If use the data augmentation
    '''
    Train_losses = []
    Train_accu = []
    Test_accu = []
    Test_losses = []
    if model_name == 'CNN' or model_name == 'Resnet':
        if model_name == 'CNN':
            model = CNN()
        else:
            model = Resnet()
        train_loader = CIFAR10(train=True, reshape=False, batch_size=batch_size, shuffle=True, reduced=reduced, enhanced=enhanced)
        val_loader = CIFAR10(train=False, reshape=False, batch_size=batch_size, shuffle=False, reduced=reduced, enhanced=enhanced)
    elif model_name == 'RNN':
        model = RNN()
        train_loader = CIFAR10(train=True, reshape=True, batch_size=batch_size, shuffle=True, reduced=reduced, enhanced=enhanced)
        val_loader = CIFAR10(train=False, reshape=True, batch_size=batch_size, shuffle=False, reduced=reduced, enhanced=enhanced)
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch, Train_losses, Train_accu)
        val(model, val_loader, epoch, Test_losses, Test_accu)
    plot_loss(Train_losses, Test_losses, reduced)
    plot_acc(Train_accu, Test_accu, reduced)


if __name__ == '__main__':
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    main(learning_rate=1e-2,model_name='RNN',reduced=True,enhanced=True,epochs=100)