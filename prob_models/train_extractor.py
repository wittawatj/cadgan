#!/usr/bin/env python3
""" Feature Extraction, using multiplabel classification using MS-COCO stuff dataset with 150 natural classes.
"""
import datetime
import argparse
import os
from math import log10
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from kbrgan.net.extractor import ResNet18_365Layer
import torchvision
from coco_dataset import Dataset_CocoSegmented
import torch.nn as nn
from tensorboardX import SummaryWriter

def to_variable(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm_clamp(x):
    return x.clamp(0, 1)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MultiLabel Classification')
parser.add_argument('--feat_size', type=int, default=64, help="number of feature channels. Default=64")
parser.add_argument('--cropsize', type=int, default=256, help="image cropsize. Default=128")
parser.add_argument('--batch_size', type=int, default= 32, help='training batch size. Default=16')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for. Default=100')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--segmented_input', type=int, default=1, help='segment input images with pretrained semantic segmentation model before feeding into G. Default=0')
parser.add_argument('--loss_mse', type=float, default=1, help='weight of MSE loss. Default=1')
parser.add_argument('--cuda', type=int, default=1, help='Try to use cuda? Default=1')
parser.add_argument('--resume',default = None, help='Provide an epoch to resume training from. Default=None')
parser.add_argument('--log_dir', default = './extractor_tb_log/', help='Log dir for tensorboardX output logs')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use. Default=4')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
parser.add_argument('--annFile', type=str, default='/home/wgondal/coco_stuff/stuff_train2017.json', help='Provide MS-COCO Stuff Segmented Annotation File. Default=0')
parser.add_argument('--path_data', type=str, default='/agbs/cpr/train2017/', help='path of train images. Default=ADE20K')
opt = parser.parse_args()

cuda = False
if opt.cuda:
    if torch.cuda.is_available():
        cuda = True
        torch.cuda.manual_seed(opt.seed)
    else:
        print('===> Warning: failed to load CUDA, running on CPU!')

# ==================  Creating tensorboardX log ============================
now = datetime.datetime.now()
training_specs = 'lr_{}_bs_{}__{}_{}-{}_{}{}'.format(opt.lr, opt.batch_size,
                now.month, now.day,now.hour, now.minute, now.second)
log_folder_path = os.path.join(opt.log_dir, training_specs)
writer = SummaryWriter(log_dir=log_folder_path)

print('===> Loading datasets')
train_set = Dataset_CocoSegmented(path=opt.path_data, cropsize= opt.cropsize,
            annotationFile = opt.annFile)

print('===> Found %d training images.' % len(train_set))
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                  batch_size=opt.batch_size, shuffle=True)

print('===> Building model')
if opt.resume:
    print('Using Pre-trained Generator model')
    G = torch.load(resume)
    G.train()
else:
    G = ResNet18_365Layer(num_classes = 183)
    if cuda:
        G = G.cuda()

'''ct = 0
for child in G.children():
    ct += 1
    if ct != 3:
        for param in child.parameters():
            param.requires_grad = False'''


print('===> Building optimizer')
optimizer = optim.Adam(G.parameters(), lr=opt.lr)
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=opt.lr)

experiment = 'multilabelBCE'
saves_path = './Extractors/'+experiment+'/saves'

if not os.path.exists(saves_path):
    os.makedirs(saves_path)

l1_loss = nn.L1Loss().cuda()
l2_loss = nn.MSELoss().cuda()


#crit = nn.BCEWithLogitsLoss() # sigmoid included
crit = nn.MultiLabelSoftMarginLoss()

steps = 0
print('===> Initializing training')
def train(epoch):
    eps = 1e-4
    global steps
    for iteration, batch in enumerate(training_data_loader, 1):
        data = to_variable(batch[0])
        label = to_variable(batch[1])#.long()
        pred = G(data)
        pred.data[pred.data < eps] = eps
        pred.data[pred.data > 1 - eps] = 1 -eps
        loss = crit(pred, label)
        predict = torch.clamp(torch.round(pred), 0, 1)
        #predict = torch.sigmoid(pred) > 0.5
        #total = len(label[1])
        r = (predict == label)#.byte())
        acc = r.float().sum().item()
        acc = float(acc) / 183 # total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss ', loss.item(), steps)
        writer.add_scalar('Acc ', acc, steps)
        steps +=1
        if iteration%50 == 0:
            print('Epoch [%d], Step[%d/%d], overall_loss: %.8f, Acc: %.4f'
              %(epoch, iteration, len(training_data_loader), loss.item(), acc))


def save_checkpoint(epoch):
    G_out_path ='%s/epoch_%s.pth'%(saves_path,str(epoch))
    if not os.path.exists(os.path.dirname(G_out_path)):
        os.makedirs(os.path.dirname(G_out_path))
    torch.save(G, G_out_path)
    print("Checkpoint saved to {}".format(G_out_path))

for epoch in range(1, opt.epochs + 1):
    train(epoch)
    if epoch % 3 == 0:
        save_checkpoint(epoch)

    # Learning rate decay
    if (epoch+1)==80 or (epoch+1)==110:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.1
