"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
import os
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# Write the default config file in the user's home directory.
# This should be consistent with glo.py.
home_dir = os.path.expanduser("~")
config_dir = os.path.join(home_dir, 'cadgan_resources')
os.path.makedirs(config_dir, exist_ok=True)

config_content = f"""#--------------------------------
# configurations for experiments. End users of this package should not need to
# change these.
[experiment]

# Full path to the directory to store temporary files when running experiments.
scratch_path = /tmp/

# Full path to the directory to store experimental results.
expr_results_path = {config_dir}/results/

# Full path to the directory to store files related to a model for a particular problem.
# Inside this folder are subfolders, each having name [problem]_[model] e.g.,
# mnist_dcgan. These folders may contain, for instance, generated images, model
# files.
problem_model_path = {config_dir}/prob_models/

#--------------------------------
# Configurations related to datasets.
[data]

# Full path to the data directory. Expected to have one subfolder for each
# problem e.g., cifar10, celeba.
data_path = {config_dir}/data

[share]

# Full path to the root directory of the shared folder. This folder contains
# all resource files (e.g., data, trained models) that are released by the
# authors.
share_path = {config_dir}/cadgan_share/
"""

# with open(path.join(config_dir, "settings.ini"),'w+') as f:
#     f.write(config_content)
config_path = os.path.join(config_dir, 'settings.ini')
with open(config_path, 'w+') as f:
    f.write(config_content)

# Get the long description from the README file
with open(path.join(config_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()



setup(
    name="cadgan",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1.0",
    description="Conditional image generation with kernel Bayes' rule",
    long_description=long_description,
    # The project's main homepage.
    url="https://github.com/wittawatj/cadgan",
    # Author details
    author="Wittawat Jitkrittum, Patsorn Sangkloy, Waleed Gondal, Amit Raj",
    author_email="wittawatj@gmail.com",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    # What does your project relate to?
    keywords="GAN kernel-methods machine-learning AI",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["data", "*.ex"]),
    # See https://www.python.org/dev/peps/pep-0440/#version-specifiers
    python_requires="~= 3.5",
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["numpy",
        "autograd",
        "scipy",
        "matplotlib",
        "ganstab @ git+https://github.com/wittawatj/GAN_stability.git#egg=ganstab-0.1.0",
        "torch==0.4.1",
        "dill>=0.2.8.2",
        "pandas>=0.23.2",
        "PerceptualSimilarity @ git+https://github.com/janesjanes/PerceptualSimilarity.git#egg=PerceptualSimilarity-0.0.1",
        # "tqdm==4.23.4","googledrivedownloader",
        "requests",
        "hed @ git+https://github.com/janesjanes/hed.git#egg=hed-0.1.0"
        ],
)
