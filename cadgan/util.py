"""A module containing convenient methods for general machine learning"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys
import time
from builtins import int, object, range, zip

import dill
import numpy as np
import pandas as pd
import PerceptualSimilarity as ps
import PerceptualSimilarity.models.dist_model as dm
import PerceptualSimilarity.util.util as psutil
import torch
from future import standard_library
from past.utils import old_div
from tqdm import tqdm

standard_library.install_aliases()
__author__ = "wittawat"


class ContextTimer(object):
    """
    A class used to time an execution of a code snippet. 
    Use it with with .... as ...
    For example, 

        with ContextTimer() as t:
            # do something 
        time_spent = t.secs

    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        if self.verbose:
            print("elapsed time: %f ms" % (self.secs * 1000))


# end class ContextTimer


class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)


# end NumpySeedContext


class ChunkIterable(object):
    """
    Construct an Iterable such that each call to its iterator returns a tuple
    of two indices (f, t) where f is the starting index, and t is the ending
    index of a chunk. f and t are (chunk_size) apart except for the last tuple
    which will always cover the rest.
    """

    def __init__(self, start, end, chunk_size):
        self.start = start
        self.end = end
        self.chunk_size = chunk_size

    def __iter__(self):
        s = self.start
        e = self.end
        c = self.chunk_size
        # Probably not a good idea to use list. Waste memory.
        L = list(range(s, e, c))
        L.append(e)
        return zip(L, L[1:])


# end ChunkIterable


def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * X.dot(Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def dist2_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance **squared** matrix of size
    X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    return D2


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def is_real_num(X):
    """return true if x is a real number. 
    Work for a numpy array as well. Return an array of the same dimension."""

    def each_elem_true(x):
        try:
            float(x)
            return not (np.isnan(x) or np.isinf(x))
        except:
            return False

    f = np.vectorize(each_elem_true)
    return f(X)


def tr_te_indices(n, tr_proportion, seed=9282):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion * n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)


def subsample_ind(n, k, seed=32):
    """
    Return a list of indices to choose k out of n without replacement
    """
    with NumpySeedContext(seed=seed):
        ind = np.random.choice(n, k, replace=False)
    return ind


def subsample_rows(X, k, seed=29):
    """
    Subsample k rows from the matrix X.
    """
    n = X.shape[0]
    if k > n:
        raise ValueError("k exceeds the number of rows.")
    ind = subsample_ind(n, k, seed=seed)
    return X[ind, :]


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to the data X (n x d) and draw J points 
    from the fit. 
    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to construct 
        a new covariance matrix before drawing samples. Useful to shrink the spread 
        of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        mean_x = np.mean(X, 0)
        cov_x = np.cov(X.T)
        if d == 1:
            cov_x = np.array([[cov_x]])
        [evals, evecs] = np.linalg.eig(cov_x)
        evals = np.maximum(0, np.real(evals))
        assert np.all(np.isfinite(evals))
        evecs = np.real(evecs)
        shrunk_cov = evecs.dot(np.diag(evals ** eig_pow)).dot(evecs.T) + reg * np.eye(d)
        V = np.random.multivariate_normal(mean_x, shrunk_cov, J)
    return V


def bound_by_data(Z, Data):
    """
    Determine lower and upper bound for each dimension from the Data, and project 
    Z so that all points in Z live in the bounds.

    Z: m x d 
    Data: n x d

    Return a projected Z of size m x d.
    """
    n, d = Z.shape
    Low = np.min(Data, 0)
    Up = np.max(Data, 0)
    LowMat = np.repeat(Low[np.newaxis, :], n, axis=0)
    UpMat = np.repeat(Up[np.newaxis, :], n, axis=0)

    Z = np.maximum(LowMat, Z)
    Z = np.minimum(UpMat, Z)
    return Z


def one_of_K_code(arr):
    """
    Make a one-of-K coding out of the numpy array.
    For example, if arr = ([0, 1, 0, 2]), then return a 2d array of the form 
     [[1, 0, 0], 
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1]]
    """
    U = np.unique(arr)
    n = len(arr)
    nu = len(U)
    X = np.zeros((n, nu))
    for i, u in enumerate(U):
        Ii = np.where(np.abs(arr - u) < 1e-8)
        # ni = len(Ii)
        X[Ii[0], i] = 1
    return X


def fullprint(*args, **kwargs):
    "https://gist.github.com/ZGainsforth/3a306084013633c52881"
    from pprint import pprint
    import numpy

    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold="nan")
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def standardize(X):
    mx = np.mean(X, 0)
    stdx = np.std(X, axis=0)
    # Assume standard deviations are not 0
    Zx = old_div((X - mx), stdx)
    assert np.all(np.isfinite(Zx))
    return Zx


def outer_rows(X, Y):
    """
    Compute the outer product of each row in X, and Y.

    X: n x dx numpy array
    Y: n x dy numpy array

    Return an n x dx x dy numpy array.
    """
    return np.einsum("ij,ik->ijk", X, Y)


def randn(m, n, seed=3):
    with NumpySeedContext(seed=seed):
        return np.random.randn(m, n)


def matrix_inner_prod(A, B):
    """
    Compute the matrix inner product <A, B> = trace(A^T * B).
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    return A.reshape(-1).dot(B.reshape(-1))


def get_classpath(obj):
    """
    Return the full module and class path of the obj. For instance, 
    kgof.density.IsotropicNormal

    Return a string.
    """
    return obj.__class__.__module__ + "." + obj.__class__.__name__


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.

    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def linear_range_transform(tensor, from_range, to_range):
    """
    Linearly interpolate from the from_range into to_range.
    For example, if from_range = (0,1), and to_range=(-2, 5), then 0 is mapped
    to -2 and 1 is mapped to 5, and all the values in-between are linearly
    interpolated.

    * tensor: a numpy or Pytorch tensor, or a scalar.
    """
    fmin, fmax = from_range
    tmin, tmax = to_range
    return (tensor - fmin) / float(fmax - fmin) * (tmax - tmin) + tmin


def download_to(url, file_path):
    """
    Download the file specified by the URL and save it to the file specified
    by the file_path. Overwrite the file if exist.
    """

    # see https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    import urllib.request
    import shutil

    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def dict_to_string(d, order=[], exclude=[], entry_sep="_", kv_sep=""):
    """
    Turn a dictionary into a string by concatenating all key and values
    together, separated by specified delimiters.


    d: a dictionary. It is expected that key and values are simple literals
        e.g., strings, float, integers, not structured objects.
    order: a list of keys in the dictionary. Keys with low indices
        appear first in the output.
    exclude: a list of keys to exclude
    entry_sep: separator between key-value entries (a string)
    kv_sep: separator between a key and its value
    """
    # dict_keys = dict( (ordered_keys[i], i) for i in range(len(ordered_keys)) )
    keys = set(d.keys()) - set(exclude)
    list_out = []
    for k in order:
        if k in keys:
            entry = k + kv_sep + str(d[k])
            list_out.append(entry)
            keys.discard(k)
    # process the rest of the keys
    for k in sorted(list(keys)):
        entry = k + kv_sep + str(d[k])
        list_out.append(entry)

    s = entry_sep.join(list_out)
    return s


def clean_filename(filename, whitelist=None, replace=" "):
    import unicodedata
    import string

    # https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    if whitelist is None:
        whitelist = valid_filename_chars
    # replace spaces
    # for r in replace:
    #     filename = filename.replace(r,'_')

    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode()

    # keep only whitelisted chars
    cleaned_filename = "".join(c for c in cleaned_filename if c in whitelist)
    return cleaned_filename


def translate_keys(d, kk):
    """
    Translate all keys in the dictionary d with kk. kk is a key-to-key
    dictionary mapping a key into its new name. No effect if kk contains extra
    keys that do not exist in d.
    d is modified in-place. 

    * This implementation is not sophiticated enough to allow swapping two
    keys. 
    """
    for k, v in kk.items():
        if k in d:
            d[kk[k]] = d[k]
            if k != kk[k]:
                del d[k]


def get_df_from_log(log_path, lpips_compute=False, model=None):
    import cadgan.imutil as imutil

    out_img_path = os.path.join(log_path, "output_images/")
    feat_img_path = os.path.join(log_path, "feature_images/")
    cond_feat_path = os.path.join(log_path, "input_feature/")
    prior_img_path = os.path.join(log_path, "prior_images/")

    iteration_sort_ind = np.argsort([int(x) for x in os.listdir(out_img_path)])
    final_t = os.listdir(out_img_path)[iteration_sort_ind[-1]]
    start_t = os.listdir(out_img_path)[iteration_sort_ind[0]]

    out_imgs = imutil.load_resize_images(os.path.join(out_img_path, final_t))
    ini_imgs = imutil.load_resize_images(os.path.join(out_img_path, start_t))

    with open(os.path.join(log_path, "metadata"), "rb") as f:
        params = dill.load(f)
    assert len(out_imgs) != 0

    h, w, c = np.shape(out_imgs[0])

    assert h == w

    # load conditioned images in the same size as the output images
    cond_imgs = imutil.load_resize_images(os.path.join(log_path, "input_images/"), resize=h)

    params["cond_imgs"] = cond_imgs
    params["out_imgs"] = out_imgs
    params["ini_imgs"] = ini_imgs
    params["log_path"] = log_path

    try:
        feat_imgs = imutil.load_resize_images(os.path.join(feat_img_path, final_t))
        params["feat_imgs"] = feat_imgs
    except:
        params["feat_imgs"] = []

    try:
        cond_feat = imutil.load_resize_images(cond_feat_path)
        params["cond_feat"] = cond_feat
    except:
        params["cond_feat"] = []
    try:
        prior_init = imutil.load_resize_images(prior_img_path)
        params["prior_ini"] = prior_init
    except:
        params["prior_ini"] = []

    params["iteration"] = int(final_t)

    if lpips_compute:
        # compute lpips score for this particular input/output/prior

        in_imgs_lpips = [np.array(x * 255.0, dtype=np.uint8) for x in cond_imgs]
        out_imgs_lpips = [np.array(x * 255.0, dtype=np.uint8) for x in out_imgs]

        dist_score, dist_mean, dist_std = get_perceptual_distance(in_imgs_lpips, out_imgs_lpips, model=model)

        params["output_score"] = [dist_mean, dist_std]
        params["output_score_matrix"] = dist_score

        dist_score, dist_mean, dist_std = get_perceptual_distance(in_imgs_lpips, in_imgs_lpips, model=model)

        params["input_score"] = [dist_mean, dist_std]
        params["input_score_matrix"] = dist_score
        if len(params["prior_ini"]) != 0:
            prior_imgs_lpips = [np.array(x * 255.0, dtype=np.uint8) for x in prior_init]
            dist_score, dist_mean, dist_std = get_perceptual_distance(in_imgs_lpips, prior_imgs_lpips, model=model)
            params["prior_score"] = [dist_mean, dist_std]
            params["prior_score_matrix"] = dist_score
    else:
        params["output_score"] = []
        params["output_score_matrix"] = []
        params["prior_score"] = []
        params["prior_score_matrix"] = []
        params["input_score"] = []
        params["input_score_matrix"] = []

    df = pd.DataFrame([params])
    return df


def get_df_from_logs(log_dir, lpips_compute=False, ignore_error=True, use_gpu=True):
    # lpips_compute can take a very longggg time, only set to true if you really want it~
    df_list = []
    if lpips_compute:
        model = dm.DistModel()
        model_path = "/notebooks/psangkloy3/amit_cadgan//LPIPS/PerceptualSimilarity/weights/v0.1/squeeze.pth"
        model.initialize(model="net-lin", net="squeeze", model_path=model_path, use_gpu=use_gpu)
    else:
        model = None
    with tqdm(total=len(os.listdir(log_dir)), file=sys.stdout) as pbar:
        for i, log in enumerate(os.listdir(log_dir)):
            pbar.set_description("processed: %d" % (1 + i))
            pbar.update(1)
            try:
                df_list.append(get_df_from_log(os.path.join(log_dir, log), lpips_compute=lpips_compute, model=model))
            except Exception as e:
                # dirty hack, old log doesn't have images/metadata, eventually we shouldn't need this
                if ignore_error:
                    pass
                else:
                    logging.error(e)
                    raise

    if len(df_list) == 0:
        return []
    else:
        return pd.concat(df_list, ignore_index=True)


def get_perceptual_distance(input_image_list, output_image_list, model_path=None, model=None, use_gpu=True):

    if model is None:
        model = dm.DistModel()
        model.initialize(
            model="net-lin", net="alex", model_path="LPIPS/PerceptualSimilarity/weights/v0.1/alex.pth", use_gpu=use_gpu
        )

    dist_scores = np.zeros((len(input_image_list), len(output_image_list)))

    for i, img_i in enumerate(input_image_list):
        for j, img_o in enumerate(output_image_list):
            if type(img_i) == str and type(img_o) == str:
                ex_i = psutil.im2tensor(psutil.load_image(img_i))
                ex_o = psutil.im2tensor(psutil.load_image(img_o))
            else:
                assert np.shape(img_i) == np.shape(img_o)
                ex_i = psutil.im2tensor(img_i)
                ex_o = psutil.im2tensor(img_o)
            dist_scores[i, j] = model.forward(ex_i, ex_o)[0]

    return dist_scores, np.mean(dist_scores), np.std(dist_scores)
