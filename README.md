# kbrgan

Conditional image generation with kernel Bayes' rule, and kernel moment
matching.

* Use Python 3.x. Pytorch 0.4.0. Tensorflow (with GPU).

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

    
* You will need to change values in `settings.ini`.

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

## Dependencies

* Parts of this repository depend on code from
  https://github.com/LMescheder/GAN_stability. To make it easier to manage dependencies, use the forked version here: https://github.com/wittawatj/GAN_stability . To install,

    1. Clone the repository
    2. `cd` to the folder. Run `pip install -e .`. 
    3. In a Python shell, you should be able to `import ganstab` without any error.

## Sharing resource files among collaborators

Generally it is not a good idea to push large files (e.g., trained GAN models)
to this repository. Since git maintains all the history, the size of the
repository can get large quickly. For sharing non-text-file resources (e.g.,
GAN models, a large collection of sample images), we will use Google Drive.
I recommend a command-line client called `drive` which can be found
[here](https://github.com/odeke-em/drive). If you use a commonly used Linux
distribution, see [this
page](https://github.com/odeke-em/drive/blob/master/platform_packages.md) for
installation instructions.
If you cannot use `drive` or prefer not to, you can use other clients that you
like (for instance, [the official
client](https://www.google.com/drive/download/)). You can also just go with
manual downloading of all the shared files from
the web, and saving them to your local directory. A drawback of this approach
is that, when we update some files, you will need to manually update them. With
the `drive` client, you simply run `drive pull` to get the latest update
(`drive` does not automatically synchronize in realtime. Other clients might.). 

To have access to our shared Google Drive folder:

1. Ask Wittawat to share with you the folder on Google Drive. You will need a
   Google account. Make sure you have a write access so you can push your
   files. Once shared, on [your Google Drive page](https://drive.google.com),
   you should see a folder called `condgan_share` on the "Shared with me" tab.
   This folder contains all the resource files (not source code) related to
   this project. Move it to your drive so that you can sync later by right
   clicking, and selecting "Add to my drive".

2. On your local machine, create a parent folder anywhere to contain all
   contents on your Google Drive (e.g., `~/Gdrive/`). We will refer to this
   folder as `Gdrive/`. Assume that you use the `drive` client. `cd` to this
   folder and run `drive init` to mark this folder as the root folder for your
   Google Drive.
   
   To get the contents in `condgan_share`: 
   
   1. Create a subfolder `Gdrive/condgan_share/`.
   2. `cd` to this subfolder and run `drive pull`. This will pull all contents 
   from the remote `condgan_share` folder to your local folder.

3. In `settings.ini` (in this repository), modify the value of the `share_path`
   key to point to your local folder `Gdrive/condgan_share/`. 

* Make sure to do `drive pull` often to get the latest update.

* After you make changes or add files, run `drive push` under `condgan_share`
   to push the contents to the remote folder for other collaborators to see.

        


