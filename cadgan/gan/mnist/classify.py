"""
Script and module to train a conv net classifier for MNIST
Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function

import argparse
import os

import cadgan
import cadgan.glo as glo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from google_drive_downloader import GoogleDriveDownloader as gdd


class MnistClassifier(nn.Module):
    def __init__(self,load=False):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        if load:
            self.download_pretrain()
            

    def forward(self, x):
        x = self.features(x)
        return F.log_softmax(x, dim=1)

    def features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def save(self, f):
        """
        Save the state of this model to a file.
        """
        torch.save(self.state_dict(), f)

    def load(self, f, **opt):
        """
        Load a Generator from a file. To be used with save().
        """
        import collections
        loaded = torch.load(f, **opt)
        if type(loaded) == collections.OrderedDict:
            return self.load_state_dict(loaded, strict=False)
        return self.load_state_dict(loaded.state_dict(), strict=False)

    def download_pretrain(self,output='',**opt):
        if output=='':
            output=glo.prob_model_folder('mnist_cnn/mnist_cnn_ep40_s1.pt')
            
        if not os.path.exists(output):
            gdd.download_file_from_google_drive(file_id='1wYJX_w3J5Fzxc5E4DCMPunWRKikLvk5F',dest_path=output)
        use_cuda = True and torch.cuda.is_available()
        load_options = {} if use_cuda else {'map_location': lambda storage, loc: storage} 
            
        self.load(output, **load_options)

# --------------


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        help="path to the file to write the trained model. If not specified, automatically name the file name with paraemters used to train. Save in the current folder.",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.save_to:
        args.save_to = "mnist_cnn_ep{}_s{}.pt".format(args.epochs, args.seed)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    data_folder = glo.data_folder()
    mnist_folder = os.path.join(data_folder, "mnist")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            mnist_folder,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            mnist_folder,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = MnistClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    # save the model
    model.save(args.save_to)


if __name__ == "__main__":
    main()
