import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../mnist_project_2/Datasets/'
use_mini_dataset = True

batch_size = 100
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.dropout = nn. Dropout(p = 0.5)

        self.conv1_1 = nn.Conv2d(1, 20, (3, 3), padding='same')
        self.conv1_2 = nn.Conv2d(20, 400, (3, 3), padding='same')
        self.l1 = nn.Linear(400 * 70, 128)
        self.o1 = nn.Linear(128, 10)

        self.conv2_1 = nn.Conv2d(1, 20, (3, 3), padding='same')
        self.conv2_2 = nn.Conv2d(20, 400, (3, 3), padding='same')
        self.l2 = nn.Linear(400 * 70, 128)
        self.o2 = nn.Linear(128, 10)

    def forward(self, x):

        o_common = self.conv1_1(x)
        o_common = F.relu(o_common)
        o_common = self.max_pool(o_common)
        o_common = self.conv2_2(o_common)
        o_common = F.relu(o_common)
        o_common = self.max_pool(o_common)
        o_common = self.flatten(o_common)
        o_common = self.dropout(o_common)

        o1 = self.l1(o_common)
        o1 = self.o1(o1)

        o2 = self.l2(o_common)
        o2 = self.o2(o2)

        return o1, o2
    

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # We need to rehape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, img_rows, img_cols))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, img_rows, img_cols))

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
