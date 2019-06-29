"""Module containing kernel related classes"""
from __future__ import division

from abc import ABCMeta, abstractmethod
from builtins import object, str

import numpy as np
import tensorflow as tf
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
class TFKernel(Kernel):
    """
    An abstract class for a kernel for Tensorflow.
    Subclasses implementing this should rely on only operations which are
    compatible with Tensorflow.
    """

    pass


# end PTKernel


class TFExplicitKernel(TFKernel):
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
        K = tf.matmul(FX, FY, transpose_b=True)
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
        vec = tf.reduce_sum(FX * FY, 1)
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
class TFKFuncCompose(TFKernel):
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
class TFKPoly(TFKernel):
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
        return (tf.matmul(X, Y, transpose_b=True) + self.c) ** self.d

    def pair_eval(self, X, Y):
        return (tf.reduce_sum(X * Y, 1) + self.c) ** self.d


# end class PTKPoly
class TFKLinear(TFKernel):
    """
    Linear kernel. Pytorch implementation.
    """

    def __init__(self):
        pass

    def eval(self, X, Y):
        return tf.matmul(X, Y, transpose_b=True)

    def pair_eval(self, X, Y):
        return tf.reduce_sum(X * Y, 1)


# end class TFKLinear
class TFKIMQ(TFKernel):
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
        sumx2 = tf.reshape(tf.reduce_sum(X ** 2, 1), [-1, 1])
        sumy2 = tf.reshape(tf.reduce_sum(Y ** 2, 1), [1, -1])
        with tf.control_dependencies(
            [tf.assert_non_negative(sumx2, name="sumx2_nonneg"), tf.assert_non_negative(sumy2, name="sumy2_nonneg")]
        ):
            D2 = sumx2 - 2.0 * tf.matmul(X, Y, transpose_b=True) + sumy2

        D2_no0 = tf.maximum(0.0, D2)
        with tf.control_dependencies([tf.assert_non_negative(D2_no0, name="D2_nonneg")]):
            K = (c ** 2 + D2_no0) ** b
        return K

    def mean_eval(self, X, Y):
        return tf.reduce_mean(self.eval(X, Y))

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c ** 2 + tf.reduce_sum((X - Y) ** 2, 1)) ** b
