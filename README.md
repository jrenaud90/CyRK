# CyRK
<div style="text-align: center;">
<a href="https://doi.org/10.5281/zenodo.7093266"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7093266.svg" alt="DOI"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12|3.13-blue" alt="Python Version 3.9-3.13" /></a>
<!-- <a href="https://codecov.io/gh/jrenaud90/CyRK" ><img src="https://codecov.io/gh/jrenaud90/CyRK/branch/main/graph/badge.svg?token=MK2PqcNGET" alt="Code Coverage"/></a> -->
<br />
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_win.yml/badge.svg?branch=main" alt="Windows Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_mac.yml/badge.svg?branch=main" alt="MacOS Tests" /></a>
<a href="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml"><img src="https://github.com/jrenaud90/CyRK/actions/workflows/push_tests_ubun.yml/badge.svg?branch=main" alt="Ubuntu Tests" /></a>
<a href="https://app.readthedocs.org/projects/cyrk/builds/?version__slug=latest"><img src="https://app.readthedocs.org/projects/cyrk/badge/?version=latest&style=flat" alt="Ubuntu Tests" /></a>


<br />
<a href="https://pypi.org/project/CyRK/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/CyRK?label=PyPI%20Downloads" /></a>
<a href="https://anaconda.org/conda-forge/cyrk"> <img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/cyrk" /> </a>
</div>

---
[Documentation](https://cyrk.readthedocs.io/en/latest/) | [GitHub](https://github.com/jrenaud90/cyrk)

<a href="https://github.com/jrenaud90/CyRK/releases"><img src="https://img.shields.io/badge/CyRK-0.15.0 Alpha-orange" alt="CyRK Version 0.15.0 Alpha" /></a>

**Runge-Kutta ODE Integrator Implemented in Cython and Numba**

CyRK provides fast integration tools to solve systems of ODEs using an adaptive time stepping scheme. CyRK can accept differential equations that are written in pure Python, njited numba, or cython-based cdef functions. These kinds of functions are generally easier to implement than pure c functions and can be used in existing Python software. Using CyRK can speed up development time while avoiding the slow performance that comes with using pure Python-based solvers like SciPy's `solve_ivp`.

The purpose of this package is to provide some 
functionality of [scipy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) with greatly improved performance.

Currently, CyRK's [numba-based](https://numba.discourse.group/) (njit-safe) implementation is **8--60x faster** than scipy's solve_ivp function.
The [cython-based](https://cython.org/) `pysolve_ivp` function that works with python (or njit'd) functions is **10-40x faster** than scipy.
The [cython-based](https://cython.org/) `cysolver_ivp` function that works with cython-based cdef functions is **40-300+x faster** than scipy.

An additional benefit of the two cython implementations is that they are pre-compiled. This avoids most of the start-up performance hit experienced by just-in-time compilers like numba.


<img style="text-align: center" src="https://github.com/jrenaud90/CyRK/blob/main/Benchmarks/CyRK_SciPy_Compare_predprey_v0-15-0.png" alt="CyRK Performance Graphic" />

## Installation

*CyRK has been tested on Python 3.9--3.13; Windows, Ubuntu, and MacOS.*

Install via pip:

`pip install CyRK`

conda:

`conda install -c conda-forge CyRK`

mamba:

`mamba install cyrk`

If not installing from a wheel, CyRK will attempt to install `Cython` and `Numpy` in order to compile the source code. A "C++ 20" compatible compiler is required.
Compiling CyRK has been tested on the latest versions of Windows, Ubuntu, and MacOS. Your milage may vary if you are using a older or different operating system.
If on MacOS you will likely need a non-default compiler in order to compile the required openMP package. See the "Installation Troubleshooting" below. 
After everything has been compiled, cython will be uninstalled and CyRK's runtime dependencies (see the pyproject.toml file for the latest list) will be installed instead.

A new installation of CyRK can be tested quickly by running the following from a python console.
```python
from CyRK import test_pysolver, test_cysolver, test_nbrk
test_pysolver()
# Should see "CyRK's PySolver was tested successfully."
test_cysolver()
# Should see "CyRK's CySolver was tested successfully."
test_nbrk()
# Should see "CyRK's nbrk_ode was tested successfully."
```

### Installation Troubleshooting

*Please [report](https://github.com/jrenaud90/CyRK/issues) installation issues. We will work on a fix and/or add workaround information here.*

- If you see a "Can not load module: CyRK.cy" or similar error then the cython extensions likely did not compile during installation. Try running `pip install CyRK --no-binary="CyRK"` 
to force python to recompile the cython extensions locally (rather than via a prebuilt wheel).

- On MacOS: If you run into problems installing CyRK then reinstall using the verbose flag (`pip install -v .`) to look at the installation log. If you see an error that looks like "clang: error: unsupported option '-fopenmp'" then you are likely using the default compiler or other compiler that does not support OpenMP. Read more about this issue [here](https://github.com/facebookresearch/xformers/issues/157) and the steps taken [here](https://github.com/jrenaud90/CyRK/blob/main/.github/workflows/push_tests_mac.yml). A fix for this issue is to use `llvm`'s clang compiler. This can be done by doing the following in your terminal before installing CyRK.
```
brew install llvm
brew install libomp

# If on ARM64 (Apple Silicon) then do:
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
# Otherwise change these directories to:
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++

pip install CyRK --no-binary="CyRK"
```

- CyRK has a number of runtime status codes which can be used to help determine what failed during integration. Learn more about these codes [https://github.com/jrenaud90/CyRK/blob/main/docs/Status%20and%20Error%20Codes.md](here).

### Development and Testing Dependencies

If you intend to work on CyRK's code base you will want to install the following dependencies in order to run CyRK's test suite and experimental notebooks.

`conda install pytest scipy matplotlib jupyter`

`conda install` can be replaced with `pip install` if you prefer.

## Using CyRK
To learn how to use CyRK, please reference our detailed documentation on Read the Docs!

[Read the Docs: CyRK Documentation](https://cyrk.readthedocs.io/en/latest/)

## Limitations and Known Issues

- [Issue 30](https://github.com/jrenaud90/CyRK/issues/30): CyRK's cysolve_ivp and pysolve_ivp does not allow for complex-valued dependent variables. 

## Citing CyRK

It is great to see CyRK used in other software or in scientific studies. We ask that you cite back to CyRK's [GitHub](https://github.com/jrenaud90/CyRK) website so interested parties can learn about this package.

It would also be great to hear about the work being done with CyRK and add your project to the list below, so get in touch!

Renaud, Joe P. (2022). CyRK - ODE Integrator Implemented in Cython and Numba. Zenodo. https://doi.org/10.5281/zenodo.7093266

In addition to citing CyRK, please consider citing SciPy and its references for the specific Runge-Kutta model that was used in your work. CyRK is largely an adaptation of SciPy's functionality. Find more details [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

## Projects using CyRK
_Don't see your project here? Create a GitHub issue or otherwise get in touch!_
- [TidalPy](https://github.com/jrenaud90/TidalPy)
- [Rydiqule](https://github.com/QTC-UMD/rydiqule)
- [KG-simulation](https://github.com/w-smialek/KG-simulation)
- [twpaSolver](https://github.com/twpalab/twpasolver)
- [NOrbit](https://github.com/curritod/norbit)
- [APPS-ODE](https://github.com/jiangnanhugo/APPS-ODE)

## Contribute to CyRK
_Please look [here](https://github.com/jrenaud90/CyRK/graphs/contributors) for an up-to-date list of contributors to the CyRK package._

CyRK is open-source and is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license. You are welcome to fork this repository and make any edits with attribution back to this project (please see the `Citing CyRK` section).
- We encourage users to report bugs or feature requests using [GitHub Issues](https://github.com/jrenaud90/CyRK/issues).
- If you would like to contribute but don't know where to start, check out the [good first issue](https://github.com/jrenaud90/CyRK/labels/good%20first%20issue) tag on GitHub.
- Users are welcome to submit pull requests and should feel free to create them before the final code is completed so that feedback and suggestions can be given early on.
