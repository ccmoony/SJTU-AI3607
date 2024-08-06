import pickle
import numpy as np
from jittor.dataset.dataset import Dataset

def load_data(is_Train=True, reshape=False, reduced=False, enhanced=True):
    if is_Train:
        image , label = [], []
        for i in range(1,6):
            with open(f'cifar-10-batches-py/data_batch_{i}', 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                image.append(np.array(data[b'data'],dtype=np.float32)/255.0)
                label.append(data[b'labels'])
        image = np.concatenate(image,axis=0).reshape(-1,3,32,32)  # shape (size,3,32,32)
        label = np.concatenate(label,axis=0)
        image_reduced = []
        label_reduced = []
        if reduced:
            for value in range(10):
                index  = np.where(label == value)[0]
                if value<5:
                    remain_size = index.shape[0]//10
                    index = np.random.choice(index,remain_size,replace=False)
                image_reduced.append(image[index])
                label_reduced.append(label[index])
            image = np.concatenate(image_reduced,axis=0)
            label = np.concatenate(label_reduced,axis=0)
        if enhanced:
            image, label = enhance_data(image, label)
    else:
        with open('cifar-10-batches-py/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            image = np.array(data[b'data'],dtype=np.float32)/255
            image = image.reshape(-1,3,32,32)      
            label = data[b'labels']
    if reshape:
        image = image.reshape(-1,96,32)
    return image, label

def enhance_data(image, label):
    image_enhanced = []
    label_enhanced = []
    for value in range(10):
        index  = np.where(label == value)[0]
        image_enhanced.append(image[index])
        label_enhanced.append(label[index])
        if value<5:
            image_enhanced.append(np.flip(image[index],axis=2))
            label_enhanced.append(label[index])
            for scale in range(1,5):
                image_enhanced.append(np.flip(image[index],axis=2)+0.001*scale*np.random.randn(*image[index].shape))
                label_enhanced.append(label[index])
                image_enhanced.append(image[index]+0.001*scale*np.random.randn(*image[index].shape))
                label_enhanced.append(label[index])
    image = np.concatenate(image_enhanced,axis=0)
    label = np.concatenate(label_enhanced,axis=0)
    return image, label

class CIFAR10(Dataset):
    def __init__(self, train=True, reshape=False, batch_size=1, shuffle=False, reduced=False, enhanced=True):
        super().__init__()
        self.is_train = train
        self.batch_size = batch_size
        self.image, self.label = load_data(train, reshape, reduced, enhanced)
        self.total_len = self.image.shape[0]
        self.shuffle = shuffle
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len, shuffle = self.shuffle)
    

    def __getitem__(self, index):
        return self.image[index], self.label[index]

