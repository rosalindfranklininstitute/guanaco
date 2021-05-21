# guanaco
> Python library for doing 3D CTF correction

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jmp1985/guanaco.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jmp1985/guanaco/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/jmp1985/guanaco.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jmp1985/guanaco/alerts/)
[![Building](https://github.com/rosalindfranklininstitute/guanaco/actions/workflows/python-package.yml/badge.svg)](https://github.com/rosalindfranklininstitute/guanaco/actions/workflows/python-package.yml)
[![Publishing](https://github.com/rosalindfranklininstitute/guanaco/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/guanaco/actions/workflows/python-publish.yml)
[![DOI](https://zenodo.org/badge/337997172.svg)](https://zenodo.org/badge/latestdoi/337997172)

## Installation

In order to build this package, the following dependencies are required:

- The CUDA toolkit
- FFTW

To install from the github repository do the following

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install git+https://github.com/rosalindfranklininstitute/guanaco.git@master
```

To install from source, clone this repository. The repository has a submodule
for pybind11 so after cloning the repository run

```sh
git submodule update --init --recursive
```

Then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .
```

If you would like to run the tests then, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .[test]
```

## Installation for developers

To install for development, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install -e .
```

## Testing

To run the tests, follow the installation instructions for developers and then do the following:

```sh
pytest
```

## Usage

To do a tomographic reconstruction with no CTF correction do something do the following

```sh
guanaco -i images.mrc -o rec.mrc -d gpu
```

To correct all images with the same single defocus something do the following

```sh
guanaco -i images.mrc -o rec.mrc -d gpu --df=20000 --Cs=2.7
```

To correct all images with the same defocus range something do the following

```sh
guanaco -i images.mrc -o rec.mrc -d gpu --df=20000 --Cs=2.7 --ndf=10
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/guanaco/issues) to submit bugs or request features.

## License

Copyright Diamond Light Source and Rosalind Franklin Institute, 2021

Distributed under the terms of the GPLv3 license, guanaco is free and open source software.

