# Content Addressable GAN (CADGAN)

Repository containing resources from our paper:

    Kernel Mean Matching for Content Addressability of GANs
    Wittawat Jitkrittum,*, Patsorn Sangkloy,* Muhammad Waleed Gondal, Amit Raj, James Hays, Bernhard Schölkopf
    ICML 2019
    (* Equal contribution)
    https://arxiv.org/abs/1905.05882

* Full paper: main text + supplement [on arXiv](https://arxiv.org/abs/1905.05882) (file size: 36MB)
* Main text only [here](http://wittawat.com/assets/papers/cadgan_icml2019_main.pdf) (file size: 7.3MB)
* Supplementary file only [here](http://wittawat.com/assets/papers/cadgan_icml2019_supp.pdf) (file size: 32MB)

We propose a novel procedure which adds *content-addressability* to any given
unconditional implicit model e.g., a generative adversarial network (GAN). The
procedure allows users to control the generative process by specifying a set
(arbitrary size) of desired examples based on which similar samples are
generated from the model. The proposed approach, based on kernel mean matching,
is applicable to any generative models which transform latent vectors to
samples, and does not require retraining of the model. Experiments on various
high-dimensional image generation problems (CelebA-HQ, LSUN bedroom, bridge,
tower) show that our approach is able to generate images which are consistent
with the input set, while retaining the image quality of the original model. To
our knowledge, this is the first work that attempts to construct, *at test
time*, a content-addressable generative model from a trained marginal model.


## Examples

We consider a GAN model from [Mescheder et al., 2018](https://arxiv.org/abs/1801.04406https://arxiv.org/abs/1801.04406) pretrained on CelebA-HQ. We run our proposed procedure using the three images (with border) at the corners as the input. All images in the triangle are the output from our procedure. Each of the output images is positioned such that the closeness to a corner (an input image) indicates the importance (weight) of the corresponding input image.

<img src="https://github.com/wittawatj/cadgan/blob/master/illus/m3_triangle_interpolation_v2.png" width="70%">

<!--<img src="https://github.com/wittawatj/cadgan/blob/master/illus/m3_triangle_interpolation_v5.png" width="70%">-->

## Demo

For a simple demo example on MNIST, check out this [Colab
notebook](https://colab.research.google.com/drive/1gH2naGOwxYNz6OGDydc9SPz7AHJlc5u7). No local installation is required.

## Code

* Support python 3.6+

* `cadgan` is intended to be a Python module i.e., it can be imported in
  Python code. Reusable code should be put in this folder. Make subfolders
  (packages) as appropriate.

* This repo is set up so that once you clone, you can do

        pip install -e /path/to/the/folder/of/this/repo/

  to install as a Python package. In Python, we can then do `import cadgan as
  cdg`, and all the code in `cadgan` folder is accessible through `cdg`.

  * Automatic dependency resolution only works with a new version of pip.
      First upgrade you pip with `pip install --upgrade pip`.

* `ipynb` folder is for Jupyter notebook files.

## Dependency, code structure, sharing resource files

You will need to change values in `settings.ini`.  See
https://github.com/wittawatj/cadgan/wiki . We currently share large files
(e.g., model files) via Google Drive.

We provide an example script to run CADGAN in `ex/run_gkmm.py`

* In case you want to experiment with the parameters, we use `ex/cmd_gkmm.py` to generate commands for multiple combinations of parameters. This requires `cmdprod` package available here: https://github.com/wittawatj/cmdprod

## Contact
If you have questions or comments, please contact [Wittawat](http://wittawat.com/) and [Patsorn](https://www.cc.gatech.edu/~psangklo/)

## TODO list
- [ ] support running cadgan on celebaHQ
- [ ] support running cadgan on LSUN
- [ ] clean up code & readme
