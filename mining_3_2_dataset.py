import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

class CIFAR100Dataset(Dataset):
    def __init__(self, transform=None, train=False):
        if train:
            sub_path = 'train'
        else:
            sub_path = 'test'
        with open(os.path.join('cifar-100-python', sub_path), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
            self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)/ 255.0
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)/ 255.0
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)/ 255.0
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)

        self.image = torch.tensor(image)
        self.label = torch.tensor(label)
        return image, label

if __name__ == "__main__":

    dataset = CIFAR100Dataset()
    print(len(dataset))

    for item in dataset:
        print(item)

# import tensorflow as tf
# class CIFAR100Dataset(Dataset):
    #     def init(self):
    #         (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
    #         self.train = torch.tensor(train_x)
    #         self.train_label = torch.tensor(train_y)
    #         self.test = torch.tensor(test_x)
    #         self.test_label = torch.tensor(test_y)
    #         self.train = self.train.reshape(-1, 32, 32, 3) / 255.0
    #         self.test = self.test.reshape(-1, 32, 32, 3) / 255.0
    #
    #         self.train = tf.convert_to_tensor(self.train.numpy(), dtype=tf.float32)
    #         self.train_label = tf.convert_to_tensor(self.train_label.numpy(), dtype = tf.float32)
    #         self.test = tf.convert_to_tensor(self.test.numpy(), dtype=tf.float32)
    #         self.test_label = tf.convert_to_tensor(self.test_label.numpy(), dtype = tf.float32)
    #
    #     def __getitem__(self, index):
    #         return self.train[index], self.train_label[index], self.test[index], self.test_label[index]
    #
    #     def len(self):
    #         return len(self.train_label)
    #
    # if __name__ == "__main__":
    #
    #     dataset = CIFAR100Dataset()
    #     print(len(dataset))
    #
    #     for item in dataset:
    #         print(item)


