import pickle
import numpy as np
from jittor.dataset.dataset import Dataset
import matplotlib.pyplot as plt

def load_data(phase='Train'):
    if phase == 'Train':
        image = []
        for i in range(1,6):
            with open(f'../hw2/cifar-10-batches-py/data_batch_{i}', 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                image.append(np.array(data[b'data'],dtype=np.float32)/255.0)
        image = np.concatenate(image,axis=0).reshape(-1,3,32,32)  # shape (size,3,32,32)
    else:
        with open('../hw2/cifar-10-batches-py/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            image = np.array(data[b'data'],dtype=np.float32)/255.0
            image = image.reshape(-1,3,32,32)      
    return image


class CIFAR10(Dataset):
    def __init__(self, phase = 'Train', batch_size=1024, patch_size=(16,16), shuffle=False):
        super().__init__()
        self.phase = phase
        self.batch_size = batch_size
        self.image = load_data(phase)
        self.length = self.image.shape[0]
        self.image_permute(patch_size=patch_size)
        self.total_len = self.image.shape[0]
        self.shuffle = shuffle
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len, shuffle = self.shuffle)

    def image_permute(self, patch_size = (16,16)):
        img_width, img_height = self.image.shape[2], self.image.shape[3]
        patch_width, patch_height = patch_size
        width_pos_list = np.arange(0, img_width, patch_height)
        height_pos_list = np.arange(0, img_height, patch_width)
        patch_num = len(width_pos_list)*len(height_pos_list)
        img_list = []
        label_list = []
        for img in self.image:
            #new_img = np.zeros(img.shape)
            new_img = np.zeros((patch_num,3,patch_width,patch_height)) # (patch_num, 3, patch_width, patch_height)
            #label = np.zeros((patch_num,patch_num))
            label = np.arange(0, patch_num, 1)
            np.random.shuffle(label)
            for pos, new_pos in enumerate(label):
                #new_img[:,width_pos_list[i]:width_pos_list[i]+patch_width, height_pos_list[j]:height_pos_list[j]+patch_height] = img[:,width_pos:width_pos+patch_width, height_pos:height_pos+patch_height]
                height_pos, width_pos = height_pos_list[new_pos//len(width_pos_list)], width_pos_list[new_pos%len(width_pos_list)]
                new_img[pos,:]=img[:, height_pos:height_pos+patch_width, width_pos:width_pos+patch_height]
                #label[j*len(width_pos_list)+i,(height_pos//patch_height)*len(width_pos_list)+width_pos//patch_width] = 1
            img_list.append(new_img)
            label_list.append(label)
        self.image = np.stack(img_list, axis=0)
        self.label = np.stack(label_list, axis=0)

    def __getitem__(self, index):
        return self.image[index], self.label[index]
    
    def __len__(self):
        return self.length

if __name__ == '__main__':
    data = CIFAR10()
    image_test, label = data[1]
    print(len(data))
    print(label)
    print(image_test.shape)
    plt.imshow(image_test[0,:].transpose(1,2,0))
    plt.show()