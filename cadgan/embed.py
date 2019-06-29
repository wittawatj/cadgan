"""Module containing code related to kernel mean embedding, conditional
embedding, and kernel Bayes' rule."""

__author__ = "wittawat"

from abc import ABCMeta, abstractmethod

import cadgan.kernel as kernel
import numpy as np
import torch
from future.utils import with_metaclass


class KEmbedding(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a kernel mean embedding. 
    """

    @abstractmethod
    def get_kernel(self):
        raise NotImplementedError()

    @abstractmethod
    def dot(self, emb):
        """
        Compute the inner product with another KEmbedding to get a real number.
        Throw a ValueError exception if emb is not compatible with this
        embedding.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_explicit_emb(self):
        """
        Return an explicit mean embedding as a vector if available.
        Return None otherwise.
        """
        raise NotImplementedError()

    def is_explicit(self):
        """
        Return True if the mean embedding can be computed explicitly.
        This is the case with a finite-dimensional kernel function.
        """
        k = self.get_kernel()
        return k.feature_map_available()

    # def is_compatible(self, emb):
    #     """
    #     Return True if this embedding is compatible with emb, so that the
    #     inner product can be computed.
    #     """
    #     assert isinstance(emb, KEmbedding)
    #     k1 = self.get_kernel()
    #     k2 = emb.get_kernel()
    #     return k1.is_compatible(k2)


# end KEmbedding


class PTImplicitKEmb(KEmbedding):
    """
    Class representing a kernel embedding as \sum_{i=1^m} w_i k(x_i, \cdot)
    for some weights {w_i}_i and samples {x_i}_i. Pytorch implementation.
    If the the feature map of the kernel is available, using PTExplicitKEmb may
    be more efficient.
    """

    def __init__(self, k, samples, weights=None):
        """
        k: a kernel instance of PTKernel
        weights: a Pytorch array of length m. If left unspecified, use uniform
            weights i.e., 1/m.
        samples: Pytorch tensor X such that X[0,...], .. X[m-1, ..] are the m
            samples representing the embedding.
        """
        assert isinstance(k, kernel.PTKernel)
        self.k = k
        self.samples = samples
        if weights is None:
            m = self.samples.shape[0]
            weights = torch.ones(m) / float(m)
        self.weights = weights

    def get_kernel(self):
        return self.k

    def dot(self, emb):
        if not isinstance(emb, PTImplicitKEmb):
            raise ValueError("Incompatible embeddings. Cannot compute a dot product.")
        # case: compatible embeddings. Pick a kernel from one of the two.
        k = self.get_kernel()

        W1 = self.weights
        X1 = self.samples
        W2 = emb.weights
        X2 = emb.samples
        # TODO: memory efficiency can be improved here
        K12 = k.eval(X1, X2)
        output = K12.mv(W2).dot(W1)
        return output

    def get_explicit_emb(self):
        return None

    def is_explicit(self):
        return False

    # --- not in the interface ---
    def eval(self, X):
        """
        Evaluate this mean embedding (treated as a function) at a set of
        n points in X.

        X: a Pytorch tensor. X[i,..] is a point.

        Return a Pytorch vector of length n.
        """
        k = self.get_kernel()
        B = self.samples
        W = self.weights
        return k.eval(X, B).mv(W)


# end PTImplicitKEmb


class PTExplicitKEmb(KEmbedding):
    """
    Class representing a kernel embedding as just a Pytorch vector. 
    This assumes that an explicit feature map of the kernel is available so that 
    the mean embedding is just a Euclidean vector.
    """

    def __init__(self, fm, emb):
        """
        fm: a FeatureMap used to create the embedding emb.
        emb: a Pytorch vector representing the explicit embedding.
        """
        assert tuple(fm.output_shape()) == tuple(emb.shape)
        self.fm = fm
        self.emb = emb

    def get_kernel(self):
        k = kernel.PTExplicitKernel(self.fm)
        return k

    def dot(self, emb):
        if not isinstance(emb, PTImplicitKEmb):
            raise ValueError("Incompatible embeddings. Cannot compute a dot product.")
        # case: compatible embeddings. Pick a kernel from one of the two.
        k = self.get_kernel()

        W1 = self.weights
        X1 = self.samples
        W2 = emb.weights
        X2 = emb.samples
        # TODO: memory efficiency can be improved here
        K12 = k.eval(X1, X2)
        output = K12.mv(W2).dot(W1)
        return output

    def get_explicit_emb(self):
        return self.emb

    def is_explicit(self):
        return True


# end PTExplicitKEmb


class KEmbSampler(with_metaclass(ABCMeta, object)):
    """
    Abstract class for algorithms which sample from a KEmbedding object.
    """

    pass


# end KEmbSampler


def kernel_herding(emb, n_sample, fn_make_optimizer=None, n_iter=200):
    """
    Perform kernel herding (sequentially solving the kernel moment matching).
    Directly optimizing (n_sample) points.
    Return Y, a collection of (n_sample) optimized points that minimize the
    moment matching loss.

    emb: PTImplicitKEmb to sample from
    n_sample: number of points to sample
    fn_make_optimizer: a function: params -> a torch.optim.XXX optimizer. 
        A function that constructs an optimizer from a list of parameters.
    n_iter: number of iterations for optimizing each y_i 
    
    Return (Y, Y0), 
        Y: a Pytorch tensor of size n_sample x dim. Optimization result
        Y0: a Pytorch tensor of size n_sample x dim. Initial points picked
    """
    if n_sample <= 0:
        raise ValueError("n_sample must be > 0. Was {}".format(n_sample))
    if fn_make_optimizer is None:
        fn_make_optimizer = lambda params: torch.optim.RMSprop(params, lr=1e-3)

    def pick_one_row(X):
        n, d = X.shape
        return torch.tensor(X[np.random.choice(n, 1)], requires_grad=True)
        # rand_vec = np.random.randn(1, d)
        # return torch.tensor(rand_vec, requires_grad=True, dtype=torch.float)

    X = emb.samples
    k = emb.get_kernel()

    # a stack of all initial points
    Y0 = []
    # first iteration. Initialize by randomly picking a point in X.
    y1 = pick_one_row(X)
    Y0.append(y1.detach().clone())
    #     y1 = y1.unsqueeze(0)

    optimizer1 = fn_make_optimizer([y1])
    for it in range(n_iter):
        loss1 = -(2.0 * emb.eval(y1).reshape(-1) - k.eval(y1, y1).reshape(-1))
        # optimize y1
        optimizer1.zero_grad()

        # compute the gradients
        loss1.backward()
        # updates
        optimizer1.step()

    Y = torch.cat([y1], dim=0)
    for t in range(2, n_sample + 1):
        yt = pick_one_row(X)
        Y0.append(yt.detach().clone())
        # add a dimension on axis=0
        #         yt = yt.unsqueeze(0)

        optimizert = fn_make_optimizer([yt])

        # optimization loop
        for it in range(n_iter):
            # optimize the rest of y2, ...y_{n_sample}
            losst = -(
                2.0 * torch.sum(emb.eval(yt))
                - (2.0 / t) * torch.sum(k.eval(Y, yt))
                - (1.0 / t) * k.eval(yt, yt).reshape(-1)
            )
            #             print(losst.item())
            # optimize yt
            optimizert.zero_grad()
            losst.backward()
            optimizert.step()

        # Now we have yt. Add it to the current set Y
        Y = torch.cat([Y, yt], dim=0)

    assert Y.shape[0] == n_sample
    Y0 = torch.cat(Y0, 0)
    return Y, Y0
