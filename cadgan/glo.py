"""A global module containing functions for managing the project."""

import configparser
import os
import pickle

import cadgan
from future import standard_library

standard_library.install_aliases()
__author__ = "wittawat"


def get_config_path():
    """
    Return the full path to the config file.
    """
    # first priority: config file at ~/cadgan_resources/settings.ini
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, 'cadgan_resources')
    at_home_config = os.path.join(config_dir, "settings.ini")
    if os.path.exists(at_home_config):
        return at_home_config
    # If the config file in the home dir does not exist,
    # Default to (root package path)/settings.ini
    root_parent = os.path.abspath(os.path.join(get_root(), os.pardir))
    at_root_config = os.path.join(root_parent, "settings.ini")
    if os.path.exists(at_root_config):
        return at_root_config
    raise RuntimeError(
        'cadgan config file does not exist. Make one at "{}" (higher priority) or "{}"'.format(
            at_home_config, at_root_config
        )
    )


def _get_config():
    config = configparser.ConfigParser()
    config_path = get_config_path()
    config.read(config_path)
    return config


def get_root():
    """Return the full path to the root of the package"""
    return os.path.abspath(os.path.dirname(cadgan.__file__))


def result_folder(*relative_path):
    """Return the full path to the result/ folder containing experimental result 
    files"""
    C = _get_config()
    path = C["experiment"]["expr_results_path"]
    if relative_path:
        path = os.path.join(path, *relative_path)
    return path


def prob_model_folder(*relative_path):
    """
    Return the full path to the folder for storing models for specific
    problems.
    """
    C = _get_config()
    path = C["experiment"]["problem_model_path"]
    if relative_path:
        path = os.path.join(path, *relative_path)
    return path


def data_folder():
    """
    Return the full path to the data folder 
    """
    C = _get_config()
    data_path = C["data"]["data_path"]
    return data_path


def data_file(*relative_path):
    """
    Access the file under the data folder. The path is relative to the 
    data folder
    """
    dfolder = data_folder()
    return os.path.join(dfolder, *relative_path)


def share_path(*relative_path):
    """
    Return the full path to the shared resource folder.
    See (share_path) key in settings.ini.
    """
    C = _get_config()
    share_path = C["share"]["share_path"]
    if relative_path:
        share_path = os.path.join(share_path, *relative_path)
    return share_path


def pickle_load_data_file(*relative_path):
    fpath = data_file(*relative_path)
    return pickle_load(fpath)


def ex_result_folder(ex):
    """Return the full path to the folder containing result files of the specified 
    experiment. 
    ex: a positive integer. """
    rp = result_folder()
    fpath = os.path.join(rp, "ex%d" % ex)
    if not os.path.exists(fpath):
        create_dirs(fpath)
    return fpath


def create_dirs(full_path):
    """Recursively create the directories along the specified path. 
    Assume that the path refers to a folder. """
    if not os.path.exists(full_path):
        os.makedirs(full_path)


def ex_result_file(ex, *relative_path):
    """Return the full path to the file identified by the relative path as a list 
    of folders/files under the result folder of the experiment ex. """
    rf = ex_result_folder(ex)
    return os.path.join(rf, *relative_path)


def ex_save_result(ex, result, *relative_path):
    """Save a dictionary object result for the experiment ex. Serialization is 
    done with pickle. 
    EX: ex_save_result(1, result, 'data', 'result.p'). Save under result/ex1/data/result.p 
    EX: ex_save_result(1, result, 'result.p'). Save under result/ex1/result.p 
    """
    fpath = ex_result_file(ex, *relative_path)
    dir_path = os.path.dirname(fpath)
    create_dirs(dir_path)
    #
    with open(fpath, "wb") as f:
        # expect result to be a dictionary
        pickle.dump(result, f)


def ex_load_result(ex, *relative_path):
    """Load a result identified by the  path from the experiment ex"""
    fpath = ex_result_file(ex, *relative_path)
    return pickle_load(fpath)


def ex_file_exists(ex, *relative_path):
    """Return true if the result file in under the specified experiment folder
    exists"""
    fpath = ex_result_file(ex, *relative_path)
    return os.path.isfile(fpath)


def pickle_load(fpath):
    if not os.path.isfile(fpath):
        raise ValueError("%s does not exist" % fpath)

    with open(fpath, "rb") as f:
        # expect a dictionary
        result = pickle.load(f)
    return result
