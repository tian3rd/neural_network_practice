from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))

test_dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))


# Define the indices
indices = list(range(len(train_dataset))) # start with all the indices in training set

split = 50 # define the split size

# Define your batch_size
#batch_size = 

# Random, non-contiguous split
#train_idx = np.random.choice(indices, size=split, replace=False)
#np.save('./output/train_idx', train_idx)

train_idx = np.load('./train_idx.npy')


#test_indices = list(range(len(test_dataset)))
test_split = 1000
#test_idx = np.random.choice(test_indices, size=test_split, replace=False)	
test_idx = np.load('./test_idx.npy')
# Contiguous split
# train_idx = indices[:split]


# define our samplers -- we use a SubsetRandomSampler because it will return
# a random subset of the split defined by the given indices without replacement
train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, sampler = train_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.test_batch_size, sampler = test_sampler, **kwargs)


ndf = 64
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            # state size. 1 x 28 x 28
            nn.Conv2d(1, ndf * 2, 2, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.fc1 = nn.Linear(8192, 50)
        self.fc_final = nn.Linear(50, 10)


    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)

model = Net()
model.load_state_dict(torch.load('./netD_epoch_20.pth', map_location=lambda storage, loc: storage), strict = False,)
if args.cuda:
    model.cuda()

optimizer = optim.SGD([{'params': model.main.parameters()}, {'params':model.fc1.parameters(), 'lr':0.01}, {'params':model.fc_final.parameters(), 'lr':0.01}], lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    confusion = torch.zeros(10,10)
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))


        if epoch%10==1:
           for i in range(len(data)):
             actual_class = target.data[i]
             predicted_class = pred.data[i]
             confusion[actual_class][predicted_class] += 1
    print('\nTraining set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, split,
        100. * correct / split))
    if epoch%10==0:
        print("confusion training")
        print(confusion)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_test = torch.zeros(10,10)
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        for i in range(len(data)):
          actual_class = target.data[i]
          predicted_class = pred.data[i]

          confusion_test[actual_class][predicted_class] += 1
    if epoch%10 == 0:
      print("confusion test")		
      print(confusion_test)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_split,
        100. * correct /test_split))


"""
for i in range(x_test_array.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
"""

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
