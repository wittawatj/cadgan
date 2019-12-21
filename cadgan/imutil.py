"""A module containing convenient methods for image processing"""

import glob
import os

import cadgan.util as util
import numpy as np
import scipy.misc
import skimage
import skimage.util
import imageio


def numpy_image_to_float(img, out_range=(0, 1)):
    """
    Convert a numpy image (3d tensor, h x w x channels) into float format where
    the output range is as given in out_range.
    Return a numpy array.
    """
    # http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.img_as_float
    new_img = skimage.util.img_as_float(img)
    if np.any(new_img < 0):
        # the range is (-1, 1)
        assert np.all(new_img <= 1)
        assert np.all(new_img >= -1)
        # transform to the specified out_range
        new_img = util.linear_range_transform(new_img, (-1, 1), out_range)
    else:
        # the range is (0, 1)
        assert np.all(new_img <= 1)
        assert np.all(new_img >= 0)
        # transform to the specified out_range
        new_img = util.linear_range_transform(new_img, (0, 1), out_range)

    return new_img


def load_resize_images(img_dir, extensions=[".jpg", ".png"], resize=256):
    """
    Load images in the specified folder and return as a list of numpy arrays.
    Do not traverse subfolders. 
    
    img_dir: path to folder containing images (and possibly other files)
    extensions: only images with these extensions will be loaded
    """
    # dir_no_slash = os.path.normpath(img_dir)
    list_patterns = [os.path.join(img_dir, "*" + ex) for ex in extensions]
    imgs = []
    for glob_pat in list_patterns:
        for img_path in sorted(glob.glob(glob_pat)):
            try:
                loaded = load_resize_image(img_path, resize=resize)
                imgs.append(loaded)
            except Exception as E:
                print(E)
                continue
    return imgs


def load_resize_image(path, resize=256):
    # img is of type imageio.core.util.Array. h x w x c
    # img = imageio.imread(path)

    # h x w x c numpy array. Range: 0-255.
    img = skimage.io.imread(path)
    # LSUN generators output 256x256 images
    if np.shape(img)[0] != resize or np.shape(img)[1] != resize:
        # skimage.transform.resize does not preserve output range. Becomes [0,
        # 1]
        img = skimage.transform.resize(img, (resize, resize), mode="reflect", anti_aliasing=True)

    # normalize the range to (0,1)
    img = numpy_image_to_float(img, (0.0, 1.0))
    return img
    # cond_imgs = img_transform(cond_imgs).unsqueeze(0).to(device)


def save_images(images, location=""):

    # images: torch tensor of size nxcxhxw where c is either 1 (grey scale) or 3 (rgb)
    if not os.path.exists(location):
        os.makedirs(location)

    np_images = images.detach().cpu().numpy()

    for i, img in enumerate(np_images):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.stack((img[:, :, 0],) * 3, axis=-1)
        fn = "%03d" % (i,)

        imageio.imwrite(os.path.join(location, fn + ".jpg"), img)
