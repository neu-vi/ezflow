import codecs
import glob
import os
import shutil

from setuptools import find_packages, setup

# Basic information
NAME = "ezflow"
DESCRIPTION = "A PyTorch library for optical flow estimation using neural networks"
VERSION = "0.2.1"
AUTHOR = "Neelay Shah"
EMAIL = "nstraum1@gmail.com"
LICENSE = "MIT"
REPOSITORY = "https://github.com/neu-vig/ezflow"
PACKAGE = "ezflow"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the keywords
KEYWORDS = ["optical flow", "pytorch", "machine learning", "deep learning"]

# Define the classifiers
# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
PKG_DESCRIBE = "README.md"

# Directories to ignore in find_packages
EXCLUDES = ()


# helper functions
def read(*parts):
    """
    returns contents of file
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as file:
        return file.read()


def get_requires(path=REQUIRE_PATH):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def get_model_zoo_configs():
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    ezflow/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs"
    )
    destination = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "ezflow",
        "model_zoo",
        "configs",
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if os.path.exists(source_configs_dir):
        if os.path.islink(destination):
            os.unlink(destination)
        elif os.path.isdir(destination):
            shutil.rmtree(destination)

    if not os.path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "configs/**/*.py", recursive=True
    )
    return config_paths


# Define the configuration
CONFIG = {
    "name": NAME,
    "version": VERSION,
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "url": REPOSITORY,
    "project_urls": {"Source": REPOSITORY},
    "packages": find_packages(
        where=PROJECT, include=["ezflow", "ezflow.*"], exclude=EXCLUDES
    ),
    "package_data": {"ezflow.model_zoo": get_model_zoo_configs()},
    "install_requires": list(get_requires()),
    "python_requires": ">=3.6",
    "test_suite": "tests",
    "tests_require": ["pytest>=3"],
    "include_package_data": True,
}

if __name__ == "__main__":
    setup(**CONFIG)
