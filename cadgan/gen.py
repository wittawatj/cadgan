"""
Module containing code to handle generative models.
"""

from __future__ import division

from abc import ABCMeta, abstractmethod
from builtins import object

import cadgan.net.net as net
import torch
import torch.nn
from future.utils import with_metaclass


class NoiseTransformer(with_metaclass(ABCMeta, torch.nn.Module)):
    """
    Class representing functions which transform random noise vectors into
    objects of interests. A Generative Adversarial Network (GAN) model is one
    such an example.
    """

    @abstractmethod
    def forward(self, *input):
        """
        Compute the output of this function given the input.
        """
        pass

    @abstractmethod
    def sample_noise(self, n, seed=1):
        """
        Sample n noise vectors from the distribution suitable for this function
        to transform. Preferably deterministic given seed.
        
        Return the n noise vectors.
        """
        pass

    @abstractmethod
    def in_out_shapes(self):
        """
        Return a tuple (I, O), where 
        * I is a tuple describing the input shape of one noise vector
        * O is a tuple describing the output shape of one transformed object.
        """
        pass

    @abstractmethod
    def sample(self, n, seed=1):
        """
        Sample n objects from this transformer. This involves sampling n noise
        vector from sample_noise(), then transforming them to genereate n
        objects. Preferably deterministic given seed.

        Return the n objects.
        """
        pass


# end class NoiseTransformer


class PTNoiseTransformer(NoiseTransformer, net.SerializableModule):
    """
    A Pytorch implementation of NoiseTransformer, meaning that 
    all input and output are given by Pytorch tensors.
    """

    pass


# end class PTNoiseTransformer


class PTNTDecPostProcess(PTNoiseTransformer):
    """
    A decorator for a PTNoiseTransformer to postprocess the output of the
    forward() function. Useful, for instance, for changing the output range
    of a GAN model from [-1, 1] to [0, 1].
    """

    def __init__(self, ptnt, module):
        """
        ptnt: an instance of PTNoiseTransformer to decorate
        module: a postprocessing module (torch.nn.Module)
        """
        super(PTNTDecPostProcess, self).__init__()
        self.ptnt = ptnt
        self.module = module

    def forward(self, *input):
        post = self.module
        ptnt = self.ptnt
        ptnt_forward = ptnt(*input)
        return post(ptnt_forward)

    def sample_noise(self, n, seed=1):
        return self.ptnt.sample_noise(n, seed)

    def in_out_shapes(self):
        return self.ptnt.in_out_shapes()

    def sample(self, n, seed=1):
        post = self.module
        sam = self.ptnt.sample(n, seed)
        return post(sam)


# end PTNTDecPostProcess


class PTNoiseTransformerAdapter(PTNoiseTransformer):
    """
    A PTNoiseTransformer whose components are specified manually as input.
    Adapter pattern.
    """

    def __init__(self, module, f_sample_noise, in_out_shapes, tensor_type=torch.cuda.FloatTensor):
        """
        ptmodule: a instance of torch.nn.Module represent a function to transform
            noise vectors.
        f_sample_noise: a function or a callable object n |-> (n x
            in_out_shapes[0] ) to sample noise vectors.
        """
        super(PTNoiseTransformerAdapter, self).__init__()
        self.module = module.eval()
        self.f_sample_noise = f_sample_noise
        self.in_out_shapes = in_out_shapes
        self.tensor_type = tensor_type

        # minimal compatibility check
        try:
            # self.sample(1)
            pass
        except:
            raise ValueError(
                "Noise sampled from f_sample_noise may be incompatible with the specified transformer module"
            )

    def forward(self, *input):
        return self.module.forward(*input)

    def sample_noise(self, n, seed=2):
        f = self.f_sample_noise
        return f(n)

    def in_out_shapes(self):
        """
        Return a tuple (I, O), where 
        * I is a tuple describing the input shape of one noise vector
        * O is a tuple describing the output shape of one transformed object.
        """
        return self.in_out_shapes

    def sample(self, n, seed=2):
        """
        Sample n objects from this transformer. This involves sampling n noise
        vector from sample_noise(), then transforming them to genereate n
        objects. 

        Return the n objects.
        """
        with torch.no_grad():
            Z = self.sample_noise(n, seed).type(self.tensor_type)
            Zvar = torch.autograd.Variable(Z)
            X = self.forward(Zvar)
        return X

    def __str__(self):
        return str(self.module)


# PTNoiseTransformerAdapter


class PTNTPostProcess(PTNoiseTransformer):
    """
    An adapter to add more modules to process the result of forward() of a
    PTNoiseTransformer.
    """

    def __init__(self, ptnt, modules):
        """
        ptnt: a PTNoiseTransformer to augment some post processing modules.
        modules: a list of torch.nn.Module's
        """
        self.ptnt = ptnt
        self.modules = modules

    def forward(self, *input, **kwargs):
        result = self.ptnt.forward(*input, **kwargs)
        for m in self.modules:
            result = m.forward(result)
        return result

    def sample_noise(self, n, seed=2):
        return self.ptnt.sample_noise(n, seed)

    def in_out_shapes(self):
        return self.ptnt.in_out_shapes()

    def sample(self, n, seed=3):
        return self.ptnt.sample(n, seed)


# PTNTPostProcess


class LinearRangeTransform(torch.nn.Module):
    """
    A Pytorch module that linearly interpolates from the from_range into to_range.
    For example, if from_range = (0,1), and to_range=(-2, 5), then 0 is mapped
    to -2 and 1 is mapped to 5, and all the values in-between are linearly
    interpolated.
    """

    def __init__(self, from_range, to_range):
        super(LinearRangeTransform, self).__init__()
        self.from_range = from_range
        self.to_range = to_range

    def forward(self, X):
        fmin, fmax = self.from_range
        tmin, tmax = self.to_range
        return (X - fmin) / float(fmax - fmin) * (tmax - tmin) + tmin


# end class LinearRangeTransform


def decode_generator(g, z0, img, f_loss, n_opt_iter=500, lr=1e-3):
    """
    Find a generator's latent noise vector z such that g(z) is close to the
    input img. The closeness is measured with f_loss.

    g: a generator of type gen.PTNoiseTransformer
    z0: a 1xlatent_dim Pytorch tensor  representing the initial noise vector
    img: one image to condition on. Minibatch size dimension is 1.
    f_loss: loss function between two images

    Return losses, Zs
        where losses is a list of loss values collected during optimiztion,
            Zs is a list of noise vectors from each iteration.
    """
    # recorded losses
    losses = []
    # recorded noise vectors
    Zs = []
    z = z0
    optimizer = torch.optim.Adam([z], lr=lr)

    for t in range(n_opt_iter):
        gen = g(z)
        pt_loss = f_loss(gen, img)
        Zs.append(z.detach().clone().cpu())
        losses.append(pt_loss.item())

        optimizer.zero_grad()
        # compute the gradients
        pt_loss.backward(retain_graph=True)
        # updates
        optimizer.step()
    return losses, Zs
