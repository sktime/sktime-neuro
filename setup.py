#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Install script for sktime-neuro"""

import codecs
import os
import platform
import re
import sys

from pkg_resources import Requirement
from pkg_resources import working_set
from setuptools import find_packages
from setuptools import setup

# raise early warning for incompatible Python versions
if sys.version_info < (3, 6):
    raise RuntimeError(
        "sktime-neuro requires Python 3.6 or higher"
        "The current Python version is %s installed in %s."
        % (platform.python_version(), sys.executable)
    )

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(HERE, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


def find_install_requires():
    """Return a list of dependencies and non-pypi dependency links.
    A supported version of tensorflow and/or tensorflow-gpu is required. If not
    found, then tensorflow is added to the install_requires list.
    Depending on the version of tensorflow found or installed, either
    keras-contrib or tensorflow-addons needs to be installed as well.
    """

    install_requires = [
        "sktime",  # ==0.6.1
        #'sktime-dl'
    ]

    return install_requires


DISTNAME = "sktime-neuro"  # package name is sktime-neuro, to have a valid module path, module name is sktime_neuro
DESCRIPTION = (
    "Neuroscience extension package for sktime, a scikit-learn "
    "compatible toolbox for learning with time series data"
)
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Tony Bagnall"
MAINTAINER_EMAIL = "ajb@uea.ac.uk"
URL = "https://github.com/sktime/sktime-neuro"
LICENSE = "BSD-3-Clause"
DOWNLOAD_URL = "https://pypi.org/project/sktime-neuro/#files"
PROJECT_URLS = {
    "Issue Tracker": "https://github.com/sktime/sktime-neuro/issues",
    "Documentation": "https://sktime.github.io/sktime-neuro/",
    "Source Code": "https://github.com/sktime/sktime-neuro",
}
VERSION = find_version("sktime_neuro", "__init__.py")
INSTALL_REQUIRES = find_install_requires()
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]

EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov" "flaky"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
    "eeg": ["mne>=0.21", "mne_bids"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
