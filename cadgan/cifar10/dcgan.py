import argparse
import math
import os
import pprint

import cadgan.cifar10.util as cifar10_util
import cadgan.gen as gen
import cadgan.glo as glo
import cadgan.log as log
import cadgan.net.net as net
import cadgan.util as util
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

# DCGAN code heavily based on
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py


class Generator(gen.PTNoiseTransformer):
    def __init__(self, latent_dim=100, f_noise=None, channels=3, minmax=(0.0, 1.0)):
        """
        f_noise: a function n |-> n x latent_dim to sample noise vectors.
            If None, use standard Gaussian.
        minmax: output range of each pixel.
        """
        raise ValueError("Not a good generator")
        if f_noise is None:
            f_noise = lambda n: torch.randn(n, latent_dim).float()
        f_noise_dim = f_noise(1).shape[1]
        if f_noise_dim != latent_dim:
            raise ValueError(
                "Inconsistent noise dimension. latent_dim={} but f_noise returns {}-dimensional vectors.".format(
                    latent_dim, f_noise_dim
                )
            )

        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.f_noise = f_noise
        self.channels = channels
        self.minmax = minmax
        img_size = 32
        self.init_size = 8

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        # tanh's range is [-1, 1]. Rescale the range to match what we want.
        minmax = self.minmax
        img = util.linear_range_transform(img, from_range=(-1, 1), to_range=minmax)
        return img

    def sample_noise(self, n):
        """
        Overriding method
        """
        return self.f_noise(n)

    def in_out_shapes(self):
        """
        Overriding method
        """
        I = self.sample(1)
        output_shape = I.shape[1:]
        return ((self.latent_dim,), output_shape)

    def sample(self, n):
        """
        Overriding method
        """
        Z = self.sample_noise(n)
        I = self.forward(Z)
        return I


# ---------


class ConvTranGenerator1(gen.PTNoiseTransformer):
    def __init__(self, latent_dim=100, f_noise=None, channels=3, minmax=(0.0, 1.0)):
        """
        f_noise: a function n |-> n x latent_dim to sample noise vectors.
            If None, use standard Gaussian.
        minmax: output range of each pixel.
        """
        if f_noise is None:
            f_noise = lambda n: torch.randn(n, latent_dim).float()
        f_noise_dim = f_noise(1).shape[1]
        if f_noise_dim != latent_dim:
            raise ValueError(
                "Inconsistent noise dimension. latent_dim={} but f_noise returns {}-dimensional vectors.".format(
                    latent_dim, f_noise_dim
                )
            )

        super(ConvTranGenerator1, self).__init__()

        self.latent_dim = latent_dim
        self.f_noise = f_noise
        self.channels = channels
        self.minmax = minmax
        img_size = 32
        self.init_size = 8

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 96, 5, 1, 0),  # 8 -> 12
            nn.BatchNorm2d(96, 0.8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 64, 9, 1, 0),  # 12 -> 20
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 7, 1, 0),  # 20 -> 26
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, channels, 7, 1, 0),  # 26 -> 32
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        # tanh's range is [-1, 1]. Rescale the range to match what we want.
        minmax = self.minmax
        img = util.linear_range_transform(img, from_range=(-1, 1), to_range=minmax)
        assert (img >= minmax[0]).all()
        assert (img <= minmax[1]).all()
        # print('output size: {}'.format(img.shape))
        return img

    def sample_noise(self, n):
        """
        Overriding method
        """
        return self.f_noise(n)

    def in_out_shapes(self):
        """
        Overriding method
        """
        I = self.sample(1)
        output_shape = I.shape[1:]
        return ((self.latent_dim,), output_shape)

    def sample(self, n):
        """
        Overriding method
        """
        Z = self.sample_noise(n)
        I = self.forward(Z)
        return I


# ---------


class ReluGenerator1(ConvTranGenerator1):
    def __init__(self, latent_dim=100, f_noise=None, channels=3, minmax=(0.0, 1.0)):
        """
        f_noise: a function n |-> n x latent_dim to sample noise vectors.
            If None, use standard Gaussian.
        minmax: output range of each pixel.
        """
        if f_noise is None:
            f_noise = lambda n: torch.randn(n, latent_dim).float()
        f_noise_dim = f_noise(1).shape[1]
        if f_noise_dim != latent_dim:
            raise ValueError(
                "Inconsistent noise dimension. latent_dim={} but f_noise returns {}-dimensional vectors.".format(
                    latent_dim, f_noise_dim
                )
            )

        super(ConvTranGenerator1, self).__init__()

        self.latent_dim = latent_dim
        self.f_noise = f_noise
        self.channels = channels
        self.minmax = minmax
        img_size = 32

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 ** 7),
            nn.LeakyReLU(),
            nn.Linear(2 ** 7, 2 ** 6),
            nn.BatchNorm1d(2 ** 6),
            nn.LeakyReLU(),
            nn.Linear(2 ** 6, 2 ** 5),
            nn.BatchNorm1d(2 ** 5),
            nn.LeakyReLU(),
            nn.Linear(2 ** 5, img_size * img_size * channels),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.net(z)
        img = out.view(out.shape[0], self.channels, 32, 32)
        # tanh's range is [-1, 1]. Rescale the range to match what we want.
        minmax = self.minmax
        img = util.linear_range_transform(img, from_range=(-1, 1), to_range=minmax)
        assert (img >= minmax[0]).all()
        assert (img <= minmax[1]).all()
        # print('output size: {}'.format(img.shape))
        return img


# end of ReluGenerator1


class PatsornGenerator1(ConvTranGenerator1):
    def __init__(self, latent_dim=100, channels=3, minmax=(0.0, 1.0)):
        """
        minmax: output range of each pixel.
        """
        super(ConvTranGenerator1, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.minmax = minmax
        img_size = 32

        # ngf = 128
        ngf = 128
        self.main = nn.Sequential(
            # input size is latent_dim
            nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(channels),
            # nn.ReLU(inplace=True),
            # state size: ngf x 32 x 32
            # nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 64 x 64
        )

    def sample_noise(self, n):
        return torch.randn(n, self.latent_dim).float()

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        img = self.main(z)
        # print('output size: {}'.format(img.shape))
        # tanh's range is [-1, 1]. Rescale the range to match what we want.
        minmax = self.minmax
        img = util.linear_range_transform(img, from_range=(-1, 1), to_range=minmax)
        # print('output size: {}'.format(img.shape))
        return img


# end of PatsornGenerator1


class SlowConvTransGenerator1(PatsornGenerator1):
    """
    Generator which has many layars of tranposed convolutions.
    Each layer only increases the size of the image slightly.
    """

    def __init__(self, latent_dim=100, channels=3, minmax=(0.0, 1.0)):
        """
        minmax: output range of each pixel.
        """
        super(ConvTranGenerator1, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.minmax = minmax
        img_size = 32

        ngf = 2 ** 6
        self.main = nn.Sequential(
            # input size is latent_dim. Start from a 1x1 image
            nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size:  4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 6, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size:  8 x 8
            nn.ConvTranspose2d(ngf * 6, ngf * 5, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 16 x 16
            nn.ConvTranspose2d(ngf * 5, ngf * 4, 4, 2, 7, bias=False),
            # nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 20 x 20
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 8, bias=False),
            # nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 26 x 26
            nn.ConvTranspose2d(ngf * 2, channels, 7, 1, 0, bias=False),
            # state size: ngf x 32 x 32
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        img = self.main(z)
        # print('output size: {}'.format(img.shape))
        # tanh's range is [-1, 1]. Rescale the range to match what we want.
        minmax = self.minmax
        img = util.linear_range_transform(img, from_range=(-1, 1), to_range=minmax)
        # print('output size: {}'.format(img.shape))
        return img


# end of SlowConvTransGenerator1


class Discriminator(net.SerializableModule):
    def __init__(self, channels=3, minmax=(0, 1.0)):
        """
        minmax: output range of each pixel.
        """
        super(Discriminator, self).__init__()
        self.minmax = minmax

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.02))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),  # 7 -> 4
            *discriminator_block(64, 128),  # 4 -> 2
        )

        self.adv_layer = nn.Sequential(nn.Linear(128 * 2 * 2, 1), nn.Sigmoid())

    def forward(self, img):
        minmax = self.minmax
        # assert (img >= minmax[0]).all()
        # assert (img <= minmax[1]).all()
        # first normalize to [-1, 1]
        img = util.linear_range_transform(img, from_range=self.minmax, to_range=(-1.0, 1.0))

        out = self.model(img)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        validity = self.adv_layer(out)
        return validity


# ---------


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_standard_normal(n, latent_dim):
    return torch.Tensor(np.random.normal(0, 1, (n, latent_dim))).float()


class DCGAN(object):
    """
    Class to manage training, model saving for DCGAN.
    """

    def __init__(
        self,
        prob_model_dir=glo.prob_model_folder("cifar10_dcgan"),
        data_dir=glo.data_file("cifar10"),
        use_cuda=True,
        n_epochs=30,
        batch_size=2 ** 6,
        lr=0.0002,
        b1=0.5,
        b2=0.999,
        latent_dim=100,
        sample_interval=200,
        save_model_interval=10,
        classes=list(range(10)),
        **op,
    ):
        """
        n_epochs: number of epochs of training
        batch_size: size of the batches
        lr: adam: learning rate
        b1: adam: decay of first order momentum of gradient
        b2: adam: decay of first order momentum of gradient
        latent_dim: dimensionality of the latent space
        sample_interval: interval between image sampling
        save_model_interval: save the generator once every this many epochs
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
        self.latent_dim = latent_dim
        self.sample_interval = sample_interval
        self.save_model_interval = save_model_interval
        self.classes = classes

    # def sample_noise(self, n):
    #     """
    #     Draw n noise vectors (input to the generator).
    #     """
    #     return torch.Tensor(np.random.normal(0, 1, (n, self.latent_dim))).float()

    @staticmethod
    def make_model_file_name(classes, epochs, batch_size):
        classes = sorted(classes)
        cls_str = "".join(map(str, classes))
        model_fname = "cifar10_c{}-dcgan-ep{}_bs{}.pt".format(cls_str, epochs, batch_size)
        return model_fname

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
        img_size = 32
        minmax = (0.0, 1.0)

        # f_noise = lambda n: sample_standard_normal(n, self.latent_dim)
        # generator = ConvTranGenerator1(latent_dim=self.latent_dim,
        #         f_noise=f_noise, channels=3,  minmax=minmax)
        # generator = ReluGenerator1(latent_dim=self.latent_dim,
        #         f_noise=f_noise, channels=3,  minmax=minmax)
        generator = PatsornGenerator1(latent_dim=self.latent_dim, channels=3, minmax=minmax)
        # generator = SlowConvTransGenerator1(latent_dim=self.latent_dim,
        #         channels=3,  minmax=minmax)
        discriminator = Discriminator(channels=3, minmax=minmax)
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
        # trdata = torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True,
        #                         transform=transforms.Compose([
        #                            transforms.ToTensor(),
        # #                            transforms.Normalize((0.1307,), (0.3081,))
        #                        ]))

        print("classes to use to train: {}".format(self.classes))
        trdata = cifar10_util.load_cifar10_class_subsets(self.classes, train=True, device="cpu", dtype=torch.float)
        print("dataset size: {}".format(len(trdata)))

        dataloader = torch.utils.data.DataLoader(trdata, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        # noise vectors for saving purpose
        z_save = generator.sample_noise(25).type(Tensor)
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
                z = Variable(generator.sample_noise(imgs.shape[0]).type(Tensor))

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
                    with torch.no_grad():
                        gev = generator.eval()
                        gen_save = gev(z_save)
                    save_image(
                        gen_save.data[:25], "%s/%06d.png" % (self.prob_model_dir, batches_done), nrow=5, normalize=False
                    )

                # keep the state of the generator
                self.generator = generator

            # Save the model once in a while
            if (epoch + 1) % self.save_model_interval == 0:
                model_fname = DCGAN.make_model_file_name(self.classes, epoch + 1, self.batch_size)
                model_fpath = os.path.join(self.prob_model_dir, model_fname)
                log.l().info("Save the generator after {} epochs. Save to: {}".format(epoch + 1, model_fpath))
                generator.save(model_fpath)


# ---------


def main():

    parser = argparse.ArgumentParser(description="Train a DCGAN on CIFAR10")
    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2 ** 5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=100,
        help="interval between image sampling. The number refers to the number of minibatch updates.",
    )
    parser.add_argument(
        "--save_model_interval", type=int, default=10, help="Save the generator once every this many epochs."
    )
    parser.add_argument("--prob_model_dir", type=str, help="interval between image sampling")
    parser.add_argument(
        "--classes", type=int, help="a list of integers (0-9) denoting the classes to consider", nargs="+"
    )

    # --------------------------------
    args = parser.parse_args()

    # op is a dict
    op = vars(args)
    if op["classes"] is None:
        # classes not specified => consider all classes
        op["classes"] = list(range(10))

    classes = sorted(op["classes"])
    cls_str = "".join(map(str, classes))
    if op["prob_model_dir"] is None:
        # use the list of classes to name the prob_model_dir
        prob_model_dir_name = "cifar10_c{}-dcgan".format(cls_str)
        op["prob_model_dir"] = glo.prob_model_folder(prob_model_dir_name)

    log.l().info("Options used: ")
    pprint.pprint(op)

    dcgan = DCGAN(**op)
    model_fname = "cifar10_c{}-dcgan-ep{}_bs{}.pt".format(cls_str, op["n_epochs"], op["batch_size"])
    model_fpath = os.path.join(dcgan.prob_model_dir, model_fname)

    # train
    log.l().info("Starting training a DCGAN on CIFAR10")
    dcgan.train()

    # save the generator
    g = dcgan.generator
    log.l().info("Saving the trained model to: {}".format(model_fpath))
    g.save(model_fpath)


if __name__ == "__main__":
    main()
