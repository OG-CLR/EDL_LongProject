'''Train CIFAR10 with PyTorch using 10 SGD configurations.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import argparse

from resnet import ResNet18
from resnet import ResNet50
from resnet import ResNet152
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config-id', type=int, default=1, help='SGD config ID (1 to 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Define the 10 SGD configurations
configs = [
    {"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0,    "nesterov": False},
    {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0,    "nesterov": False},
    {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4,   "nesterov": True},
    {"lr": 0.05, "momentum": 0.0, "weight_decay": 0.0,    "nesterov": False},
    {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.0,    "nesterov": True},
    {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4,   "nesterov": True},
    {"lr": 0.1,  "momentum": 0.0, "weight_decay": 0.0,    "nesterov": False},
    {"lr": 0.1,  "momentum": 0.9, "weight_decay": 0.0,    "nesterov": False},
    {"lr": 0.1,  "momentum": 0.9, "weight_decay": 5e-4,   "nesterov": False},
    {"lr": 0.1,  "momentum": 0.9, "weight_decay": 5e-4,   "nesterov": True},
]

config = configs[args.config_id - 1]

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
num_epochs = 200

# Data preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Use a subset of CIFAR-10
np.random.seed(2147483647)
indices = np.random.permutation(len(trainset))[:15000]
train_subset = torch.utils.data.Subset(trainset, indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = ResNet152().to(device) 
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Resume from checkpoint if needed
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/ckpt_config{args.config_id}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Optimizer and scheduler
optimizer = optim.SGD(
    net.parameters(),
    lr=config['lr'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay'],
    nesterov=config['nesterov']
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

criterion = nn.CrossEntropyLoss()

# Training and test functions
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = train_loss / (batch_idx + 1)
        progress_bar(batch_idx, len(trainloader), f'Loss: {avg_loss:.3f} | Acc: {acc:.2f}% ({correct}/{total})')

    return acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            avg_loss = test_loss / (batch_idx + 1)
            progress_bar(batch_idx, len(testloader), f'Loss: {avg_loss:.3f} | Acc: {acc:.2f}% ({correct}/{total})')

    acc = 100. * correct / total
    if acc > best_acc:
        print('📦 Saving best model...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt_config{args.config_id}.pth')
        best_acc = acc

    return acc

# Main training loop
for epoch in range(start_epoch, start_epoch + num_epochs):
    train_acc = train(epoch)
    test_acc = test(epoch)
    scheduler.step()
    print(f"✅ End of epoch {epoch + 1}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")
