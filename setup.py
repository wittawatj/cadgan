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



here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()



setup(
    name="cadgan",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1.0",
    description="Conditional image generation with kernel mean matching",
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
    python_requires="~= 3.6",
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
        "torchsummary",
        "pyyaml",
        "tensorboardX",
        "googledrivedownloader",
        "hed @ git+https://github.com/janesjanes/hed.git#egg=hed-0.1.0",
        "tensorboard"
        ],
)
