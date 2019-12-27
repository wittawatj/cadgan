"""Module containing kernel related classes"""
from __future__ import division

from abc import ABCMeta, abstractmethod
from builtins import object, str

import numpy as np
import torch
from future.utils import with_metaclass
from past.utils import old_div

__author__ = "wittawat"


class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        pass

    # @abstractmethod
    # def is_compatible(self, k):
    #     """
    #     Return True if the given kernel k is "compatible" with this kernel.
    #     The term compatible is vague. Ideally we want to be able to check that
    #     the two kernels define the same RKHS. However, this is difficult to
    #     implement without an elaborate type system.

    #     Simply check whether the kernel has the same type and the same (or
    #     approximately the same) parameters.
    #     """
    #     pass

    def get_feature_map(self):
        """
        Return the underlying feature map (an instance of FeatureMap) of the
        kernel.  Return None if a closed-form feature map is not available
        e.g., the output of the underlying feature map is infinite-dimensional.
        """
        return None

    def feature_map_available(self):
        """
        Return True if an explicit feature map is available.
        """
        return self.get_feature_map() is not None

    # @abstractmethod
    # def is_inf_dim(self):
    #     """
    #     Return true if the kernel is infinite dimensional.
    #     """
    #     pass


# end Kernel


class FeatureMap(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a feature map of a kernel.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self):
        """
        Return the expected input shape of this feature map (excluding the
        batch dimension).  For instance if each point is a 32x32 pixel image,
        then return (32, 32).
        """
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self):
        """
        Return the output shape of this feature map.
        """
        raise NotImplementedError()


# end class FeatureMap


class FuncFeatureMap(FeatureMap):
    def __init__(self, f, in_shape, out_shape):
        """
        f: a callable object representing the feature map.
        in_shape: expected shape of the input
        out_shape: expected shape of the output
        """
        self.f = f
        self.in_shape = in_shape
        self.out_shape = out_shape

    def __call__(self, x):
        f = self.f
        return f(x)

    def input_shape(self):
        return self.in_shape

    def output_shape(self):
        return self.out_shape


# end of FuncFeatureMap


class PTKernel(Kernel):
    """
    An abstract class for a kernel for Pytorch.
    Subclasses implementing this should rely on only operations which are
    compatible with Pytorch.
    """

    pass


# end PTKernel


class PTExplicitKernel(PTKernel):
    """
    A class for kernel that is defined as 
        k(x,y) = <f(x), f(y)> 
    for a finite-output f (of type FeatureMap).
    """

    def __init__(self, fm):
        """
        fm: a FeatureMap parameterizing the kernel. This feature map is
            expected to take in a Pytorch tensor as the input.
        """
        self.fm = fm

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on Pytorch tensors X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        K = FX.mm(FY.t())
        return K

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d Pytorch array of length n.
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        vec = torch.sum(FX * FY, 1)
        return vec

    # def is_compatible(self, k):
    #     """
    #     This compatibility check is very weak.
    #     """
    #     if isinstance(k, PTExplicitKernel):
    #         fm1 = self.fm
    #         fm2 = k.fm
    #         return fm1.input_shape() == fm2.input_shape() and \
    #                 fm1.output_shape() == fm2.output_shape()
    #     return False

    def get_feature_map(self):
        return self.fm


# end PTExplicitKernel


class PTKFuncCompose(PTKernel):
    """
    A kernel given by k'(x,y) = k(f(x), f(y)), where f is the specified 
    function, and k is the specified kernel.
    f has to be callable.
    """

    def __init__(self, k, f):
        """
        k: a PTKernel
        f: a callable object or a function
        """
        self.k = k
        self.f = f

    def eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f(X)
        fy = f(Y)
        return k.eval(fx, fy)

    def pair_eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f(X)
        fy = f(Y)
        return k.pair_eval(fx, fy)


# end class PTKFuncCompose


class PTKPoly(PTKernel):
    """
    Polynomial kernel of the form
    k(x,y) = (x^T y + c)^d
    """

    def __init__(self, c, d):
        if c < 0:
            raise ValueError("c has to be positive real. Was {}".format(c))
        if d < 0:
            raise ValueError("d has to be positive integer. Was {}".format(d))
        self.c = c
        self.d = d

    def eval(self, X, Y):
        return (X.mm(Y.t()) + self.c) ** self.d

    def pair_eval(self, X, Y):
        return (torch.sum(X * Y, 1) + self.c) ** self.d


# end class PTKPoly


class PTKLinear(PTKernel):
    """
    Linear kernel. Pytorch implementation.
    """

    def __init__(self):
        pass

    def eval(self, X, Y):
        return X.mm(Y.t())

    def pair_eval(self, X, Y):
        return torch.sum(X * Y, 1)


# end class PTKLinear


class PTKIMQ(PTKernel):
    """
    The inverse multiquadric (IMQ) kernel studied in 

    Measure Sample Quality with Kernels 
    Jackson Gorham, Lester Mackey

    k(x,y) = (c^2 + ||x-y||^2)^b 
    where c > 0 and b < 0. Following a theorem in the paper, this kernel is 
    convergence-determining only when -1 < b < 0. In the experiments, 
    the paper sets b = -1/2 and c = 1.
    """

    def __init__(self, b=-0.5, c=1.0):
        if not b < 0:
            raise ValueError("b has to be negative. Was {}".format(b))
        if not c > 0:
            raise ValueError("c has to be positive. Was {}".format(c))
        self.b = b
        self.c = c

    def eval(self, X, Y):
        b = self.b
        c = self.c
        sumx2 = torch.sum(X ** 2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y ** 2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0 * X.mm(Y.t()) + sumy2
        K = (c ** 2 + D2) ** b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c ** 2 + torch.sum((X - Y) ** 2, 1)) ** b


# end class PTKIMQ


class PTKL1Distance(PTKernel):
    """
    Pytorch implementation of a the kernel

        k(x,y) = exp(-||x-y||_1/sigma)

    (1-norm, not squared)
    For some positive sigma (bandwidth). This kernel is positive definite. See
    "Kernel Choice and Classifiability for RKHS Embeddings of Probability
    Distributions", NIPS 2009, page 4.

    """

    def __init__(self, sigma):
        assert sigma > 0, "sigma must be > 0. Was {}".format(sigma)
        self.sigma = sigma

    def eval(self, X, Y):
        """
        Evaluate the Laplace kernel on the two two-dimensional Pytorch tensors
        X, Y.

        * X: n1 x d Pytorch
        * Y: n2 x d Pytorch

        Return
        ------
        K: an n1 x n2 Gram matrix.
        """
        n1 = X.shape[0]
        # TODO: Improve computational efficiency
        K = torch.tensor([], device=X.device, dtype=X.dtype)
        for i in range(n1):
            D1XiY = torch.sum(torch.abs(X[i, :] - Y), 1)
            KXiY = torch.exp(-D1XiY / self.sigma)
            K = torch.cat((K, KXiY.unsqueeze(0)), 0)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D1 = torch.sum(torch.abs(X - Y), 1)
        Kvec = torch.exp(-D1 / self.sigma)
        return Kvec


# end class PTKL1Distance


class PTKLaplace(PTKernel):
    """
    Pytorch implementation of a Laplace kernel

        k(x,y) = exp(-||x-y||_2/sigma)

    (2-norm, not squared)
    For some positive sigma (bandwidth).
    """

    def __init__(self, sigma):
        assert sigma > 0, "sigma must be > 0. Was %s" % str(sigma)
        self.sigma = sigma
        raise NotImplementedError("Need to check the correctness of the implementation")

    def eval(self, X, Y):
        """
        Evaluate the Laplace kernel on the two two-dimensional Pytorch tensors
        X, Y.

        * X: n1 x d Pytorch
        * Y: n2 x d Pytorch

        Return
        ------
        K: an n1 x n2 Gram matrix.
        """
        sumx2 = torch.sum(X ** 2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y ** 2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0 * X.mm(Y.t()) + sumy2
        # remove negative values
        D2[D2 < 0] = 0
        D = D2 ** 0.5
        K = torch.exp(-D / self.sigma)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = torch.sum((X - Y) ** 2, 1)
        D2[D2 < 0] = 0
        D = D2 ** 0.5
        Kvec = torch.exp(-D / self.sigma)
        return Kvec


# end class PTKLaplace


class PTKGauss(PTKernel):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        assert sigma2 > 0, "sigma2 must be > 0. Was %s" % str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two two-dimensional Pytorch tensors
        X, Y.

        * X: n1 x d Pytorch
        * Y: n2 x d Pytorch

        Return
        ------
        K: an n1 x n2 Gram matrix.
        """
        sumx2 = torch.sum(X ** 2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y ** 2, 1).reshape(1, -1)
        D2 = sumx2 - 2.0 * X.mm(Y.t()) + sumy2
        K = torch.exp(-D2 / (2.0 * self.sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = torch.sum((X - Y) ** 2, 1)
        Kvec = torch.exp(old_div(-D2, (2.0 * self.sigma2)))
        return Kvec

    # def is_compatible(self, k):
    #     assert isinstance(k, PTKGauss)
    #     w1 = self.sigma2
    #     w2 = k.sigma2
    #     return np.abs(w1 - w2) <= 1e-6


# end class PTKGauss


class KGauss(Kernel):
    """
    The standard isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        assert sigma2 > 0, "sigma2 must be > 0. Was %s" % str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        # (n1, d1) = X.shape
        # (n2, d2) = Y.shape
        # assert d1==d2, 'Dimensions of the two inputs must be the same'
        sumx2 = np.reshape(np.sum(X ** 2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y ** 2, 1), (1, -1))
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2
        K = np.exp(old_div(-D2, (2.0 * self.sigma2)))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.exp(old_div(-D2, (2.0 * self.sigma2)))
        return Kvec

    # def is_compatible(self, k):
    #     assert isinstance(k, KGauss)
    #     w1 = self.sigma2
    #     w2 = k.sigma2
    #     return np.abs(w1 - w2) <= 1e-6

    def __str__(self):
        return "KGauss(%.3f)" % self.sigma2


# end class KGauss
