# kbrgan

Conditional image generation with kernel Bayes' rule, and kernel moment
matching.

* `kbrgan` is intended to be a Python module i.e., it can be imported in
  Python code. Reusable code should be put in this folder. Make subfolders
  (packages) as appropriate.

* Using Anaconda is the probably the easiest way to setup a Python environment.
  The file `kbrgan_conda_env.yml` is a file exported from Anaconda. It can be
  used to create (or update) a new environment for this project. We all will
  have the same running software. Use the following commands:

        // create a new environment called randgan
        conda create --name kbrgan

        // activate the environment
        conda activate kbrgan 

        // update it with the file
        conda env update -f kbrgan_conda_env.yml

    

* After activating an appropriate anaconda environment, this repo is set up so
  that once you clone, you can do 

        pip install -e /path/to/the/folder/of/this/repo/

  to install as a Python package. In Python, we can then do `import kbrgan as
  kbg`, and all the code in `kbrgan` folder is accessible through `kbg`.

* `ipynb` folder is for Jupyter notebook files. Easiest to create
  `ipynb/wittawat/`, `ipynb/waleed/`, `ipynb/amit`, and `ipynb/patsorn/`. Or you can also
  create branches if you like.

* If you feel that your code is really a standalone script, you can also create
  put your code in `script/` at the root level, if you prefer that way.


## Dependency, code structure, sharing resource files

You will need to change values in `settings.ini`.  See
https://github.com/wittawatj/kbrgan/wiki . We currently share large files
(e.g., model files) via Google Drive. 


