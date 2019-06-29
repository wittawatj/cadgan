"""
Module containing general Tensorflow code for neural networks.
For Pytorch, see net.py.
"""

from abc import ABCMeta, abstractmethod
from builtins import object

from future.utils import with_metaclass

# import numpy as np
# import tensorflow as tf


class ImgFeatureExtractor(object):
    """
    An abstract class for a feature extractor for images.
    """

    @abstractmethod
    def extract(self, imgs):
        """
        imgs: A stack of images such that imgs[i,...] is one image. 
            Each image can have more than one channel.

        Return a Tensorflow tensor of size n x d where n = imgs.shape[0],
            and d  = number of extracted features
        """
        raise NotImplementedError()

    def __call__(self, imgs):
        return self.extract(imgs)


# end ImgFeatureExtractor
