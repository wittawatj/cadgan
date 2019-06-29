"""
Module containing general Pytorch code for neural networks.
"""

from abc import ABCMeta, abstractmethod
from builtins import object

# dill (https://dill.readthedocs.io/en/latest/) is perhaps more powerful than pickle.
# It can serialize function handles (e.g., lambda).
import dill
# import numpy as np
import torch
import torch.nn
from future.utils import with_metaclass


class SerializableModule(torch.nn.Module):
    """
    A Pytorch module which can be serialized to disk.
    """

    def save(self, f):
        """
        Save the state of this model to a file.
        f can be a file handle or a string representing the file path.
        Subclasses should override this method if needed.
        """
        torch.save(self, f, pickle_module=dill)

    @staticmethod
    def load(f, **opt):
        """
        Load the module as saved by the self.save(f) method.
        Subclasses should override this static method.
        """
        return torch.load(f, pickle_module=dill, **opt)


# end class SerializableModule


class ModuleAdapter(SerializableModule):
    """
    A torch.nn.Module object whose forward() function is directly specified by 
    a callable object (e.g., function handle).
    """

    def __init__(self, f):
        """
        f: a callable object. Preferably f should be such that it can be
            serialized (i.e., pickled) safely.
        """
        super(ModuleAdapter, self).__init__()
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        f = self.f
        return f(*args, **kwargs)


# ModuleAdapter
