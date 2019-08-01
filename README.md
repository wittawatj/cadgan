# cadgan

Conditional image generation with kernel moment matching.

* [Link to Overleaf](https://www.overleaf.com/project/5c2d971d1cf80f23a8a26dcb) for ICML 2019:

* Support python 3.6+

* `cadgan` is intended to be a Python module i.e., it can be imported in
  Python code. Reusable code should be put in this folder. Make subfolders
  (packages) as appropriate.

* Using Anaconda is the probably the easiest way to setup a Python environment.
  The file `cadgan_conda_env.yml` is a file exported from Anaconda. It can be
  used to create (or update) a new environment for this project. We all will
  have the same running software. Use the following commands:

        // create a new environment called randgan
        conda create --name cadgan python=3.7

        // activate the environment
        conda activate cadgan

        // update it with the file
        conda env update -f cadgan_conda_env.yml



* After activating an appropriate anaconda environment, this repo is set up so
  that once you clone, you can do

        pip install -e /path/to/the/folder/of/this/repo/

  to install as a Python package. In Python, we can then do `import cadgan as
  cdg`, and all the code in `cadgan` folder is accessible through `cdg`.

* `ipynb` folder is for Jupyter notebook files. Easiest to create
  `ipynb/wittawat/`, `ipynb/waleed/`, `ipynb/amit`, and `ipynb/patsorn/`. Or you can also
  create branches if you like.

* If you feel that your code is really a standalone script, you can also create
  put your code in `script/` at the root level, if you prefer that way.


## Dependency, code structure, sharing resource files

You will need to change values in `settings.ini`.  See
https://github.com/wittawatj/cadgan/wiki . We currently share large files
(e.g., model files) via Google Drive.


* Need `cmdprod` package to reproduct the experimental results.
