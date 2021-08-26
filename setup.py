#!/usr/bin/env python

"""The setup script."""

import codecs
import os

from setuptools import find_packages, setup

PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
PKG_DESCRIBE = "README.md"


def read(*parts):
    """
    returns contents of file
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as file:
        return file.read()


REQUIRE_PATH = "requirements.txt"


def get_requires(path=REQUIRE_PATH):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


test_requirements = [
    "pytest>=3",
]

setup(
    author="Neelay Shah",
    author_email="nstraum1@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A PyTorch library for optical flow estimation",
    entry_points={
        "console_scripts": [
            "openoptflow=openoptflow.cli:main",
        ],
    },
    install_requires=list(get_requires()),
    license="MIT license",
    include_package_data=True,
    keywords="openoptflow",
    name="openoptflow",
    packages=find_packages(include=["openoptflow", "openoptflow.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/neu-vig/openoptflow",
    version="0.1.0",
    zip_safe=False,
)
