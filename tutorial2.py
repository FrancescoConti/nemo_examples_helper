# these should already be available

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import nemo
from tqdm import tqdm
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # <== Module, not Function!
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # <== Module, not Function!
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 256)
        self.fcrelu1 = nn.ReLU() # <== Module, not Function!
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) # <== Module, not Function!
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) # <== Module, not Function!
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fcrelu1(x) # <== Module, not Function!
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # <== the softmax operation does not need to be quantized, we can keep it as it is
        return output
# convenience class to keep track of averages
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
    def update(self, val):
        self.sum += val.cpu()
        self.n += 1
    @property
    def avg(self):
        return self.sum / self.n

def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    train_loss = Metric('train_loss')
    with tqdm(total=len(train_loader),
          desc='Train Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item()})
            t.update(1)
    return train_loss.avg.item()

def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    with tqdm(total=len(test_loader),
          desc='Test',
          disable=not verbose) as t:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc.update((pred == target.view_as(pred)).float().mean())
                t.set_postfix({'acc' : test_acc.avg.item() * 100. })
                t.update(1)
    test_loss /= len(test_loader.dataset)
    return test_acc.avg.item() * 100.
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=128, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=128, shuffle=False, **kwargs
)

def get_model():
    model = ExampleNet().to(device)
    state_dict = torch.load("mnist_cnn_fp.pt", map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    return model
