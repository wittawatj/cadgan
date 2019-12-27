import math
import os

import cadgan.glo as glo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from google_drive_downloader import GoogleDriveDownloader as gdd

# DCGAN code heavily based on
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py


class Generator(nn.Module):
    def __init__(self, latent_dim=100, load=False):
        super(Generator, self).__init__()

        img_size = 28
        channels = 1
        self.latent_dim = latent_dim
        self.init_size = img_size // 4

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        if load:
            self.download_pretrain()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def save(self, f):
        """
        Save the state of this model to a file f.
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

    def download_pretrain(self, output_path='',**opt):
        if output_path == '':
            output_path = glo.prob_model_folder('mnist_dcgan/mnist_dcgan_ep{}_bs{}.pt'.format(40,64))
        if not os.path.exists(output_path):
            gdd.download_file_from_google_drive(file_id='1KOi9b8JSBXc7hx9P8Azbtkhd7fZOeZxc',dest_path=output_path)
            
        use_cuda = torch.cuda.is_available()
        load_options = {} if use_cuda else {'map_location': lambda storage, loc: storage} 
        self.load(output_path,**load_options)
# ---------


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        channels = 1
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # size 28 -> 15
            *discriminator_block(16, 32),  # 14 -> 7
            *discriminator_block(32, 64),  # 7 -> 4
            *discriminator_block(64, 128),  # 4 -> 2
        )

        # The height and width of downsampled image
        img_size = 28
        self.adv_layer = nn.Sequential(nn.Linear(128 * 2 * 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        validity = self.adv_layer(out)
        return validity

    def save(self, f):
        """
        Save the state of this model to a file.
        """
        torch.save(self, f)

    @staticmethod
    def load(f):
        """
        Load a Generator from a file. To be used with save().
        """
        return torch.load(f)


# ---------


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# parser = argparse.ArgumentParser()
# parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
# parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
# parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
# parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')


class DCGAN(object):
    """
    Class to manage training, model saving for DCGAN.
    """

    def __init__(
        self,
        prob_model_dir=glo.prob_model_folder("mnist_dcgan"),
        data_dir=glo.data_file("mnist/"),
        use_cuda=True,
        n_epochs=30,
        batch_size=2 ** 6,
        lr=0.0002,
        b1=0.5,
        b2=0.999,
        n_cpu=4,
        latent_dim=100,
        sample_interval=400,
    ):
        """
        n_epochs: number of epochs of training
        batch_size: size of the batches
        lr: adam: learning rate
        b1: adam: decay of first order momentum of gradient
        b2: adam: decay of first order momentum of gradient
        n_cpu: number of cpu threads to use during batch generation
        latent_dim: dimensionality of the latent space
        sample_interval: interval between image sampling
        """
        os.makedirs(prob_model_dir, exist_ok=True)
        self.prob_model_dir = prob_model_dir
        self.data_dir = data_dir
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_cpu = n_cpu
        self.latent_dim = latent_dim
        self.sample_interval = sample_interval

    def sample_noise(self, n):
        """
        Draw n noise vectors (input to the generator).
        """
        return torch.Tensor(np.random.normal(0, 1, (n, self.latent_dim))).float()

    def save_state(self, f):
        """
        Save state of this object to a file.
        """
        torch.save(self, f)

    def load_state(self, f):
        """
        Load the state of a DCGAN object from a file. 
        Return a DCGAN object.
        """
        return torch.load(f)

    def train(self):
        """
        Traing a DCGAN model with the training hyperparameters as specified in
        the constructor. Directly modify the state of this object to store all
        relevant variables.

        * self.generator stores the trained generator.
        """

        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        img_size = 28
        generator = Generator(latent_dim=self.latent_dim)
        discriminator = Discriminator()

        cuda = True if torch.cuda.is_available() else False

        if self.use_cuda and cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        # Configure data loader
        os.makedirs(self.data_dir, exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        # transforms.Resize(self.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        for epoch in range(self.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.sample_noise(imgs.shape[0]).type(Tensor))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    save_image(
                        gen_imgs.data[:25], "%s/%d.png" % (self.prob_model_dir, batches_done), nrow=5, normalize=True
                    )

                # keep the state of the generator
                self.generator = generator


# ---------
