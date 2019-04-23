from neural_network import minibatch_gd, test_nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import struct
import argparse


# Our convolution neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Declare the layers of the network which have parameters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=4)
        self.fc1 = nn.Linear(in_features=50*4*4, out_features=500)
        self.fc2 = nn.Linear(500, 10)

    # Combine the layers
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50*4*4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Checks how accurate our results are compared to given data
def test(net, testLoader):
    net.eval()
    correct = 0
    once = False

    with torch.no_grad():

        # Iterates through each batch (@Jack declare ur vars outside this loop)
        for (data,target) in testLoader:
            output = net(data)
            pred = output.max(1, keepdim=True)[1]

            # TODO @Jack: Implement Confusion matrix
            # (target is an array of actual classes, pred is array of the classes we predicted)
            # Their formats are different however, target[i] gets actual class of ith element
            # pred[i][0] gets predicted class ith element. Use target.shape[0] for number of
            # elements. )

            # @Jack You can ignore this line if you want, but will get you total correct
            # good for average classification rate)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # TODO @Jack: Find the average classification rate per class (from the Confusion matrix)

        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))


# Class holding formatted data that we pass into our network
class OurDataset(Dataset):
    def __init__(self, fnData, fnLabels):
        self.LoadData(fnData)
        self.LoadLabels(fnLabels)
        assert self.l.size()[0]==self.d.size()[0]

    def LoadLabels(self, fnLabels):
        self.l = torch.LongTensor(fnLabels)

    def LoadData(self, fnData):
        res = (2051, fnData.shape[0], 28, 28)
        self.d = torch.zeros(res[1], 1, res[2], res[3])
        for i in range(fnData.shape[0]):
            tmp = torch.Tensor(fnData[i])
            tmp = tmp.view(1, res[2], res[3])

            self.d[i,:,:,:] = tmp

    def __len__(self):
        return self.d.size()[0]
    def __getitem__(self, idx):
        return (self.d[idx,:,:], self.l[idx])


if __name__ == '__main__':

    # Parse our arguments
    parser = argparse.ArgumentParser(description='CS440 MP4 Snake')
    parser.add_argument('--epochs', dest="epoch", type=int, default=5,
                    help='number of testing episodes - default 5')

    parser.add_argument('--batch_size', dest="b_size", type=int, default=128,
                    help='number of testing episodes - default 128')

    args = parser.parse_args()
    print("Training for epochs: ", args.epoch)

    # Load in raw data as numpy arrays
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    # Format our data
    trainData = OurDataset(x_train, y_train)
    testData = OurDataset(x_test, y_test)

    # Initialize our neural networks
    trainLoader = DataLoader(trainData, batch_size=args.b_size, shuffle=True, num_workers=0)
    testLoader = DataLoader(testData, batch_size=args.b_size, shuffle=False, num_workers=0)

    net = Net()

    numparams = 0
    for f in net.parameters():
        numparams += f.numel()
    print("Number of parameters trained", numparams)

    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
    optimizer.zero_grad()

    # Our loss function
    criterion = nn.CrossEntropyLoss()

    # Accuracy based on initial weights
    test(net, testLoader)

    # Repeating per epoch
    for e in range(args.epoch):
        net.train()

        # Iterating through our batches
        for batch_idx, (data, target) in enumerate(trainLoader):
            pred = net(data)
            loss = criterion(pred, target)
            loss.backward()
            gn = 0
            for f in net.parameters():
                gn = gn + torch.norm(f.grad)
            # print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
            optimizer.step()
            optimizer.zero_grad()

        # Printing out the accuracy after each epoch
        test(net, testLoader)
