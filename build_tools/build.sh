#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/install.sh

set -e

echo "Setting up conda env ..."
echo "Python version: " "$PYTHON_VERSION"

# Deactivate the any previously set virtual environment and setup a
# conda-based environment instead
deactivate || :

# Configure conda
conda config --set always_yes true
conda update --quiet conda

# Set up test environment
conda create --name testenv python="$PYTHON_VERSION"

# Activate environment
source activate testenv

# Install requirements from inside conda environment
pip install -r "$REQUIREMENTS"

# Build sktime-dl
# invokes build_ext -i to compile files
# builds universal wheel, as specified in setup.cfg
python setup.py bdist_wheel

# Install from built wheels
pip install --pre --no-index --no-deps --find-links dist/ sktime-neuro


set +e