# Installation

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

## Installation Troubleshooting

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

## Development and Testing Dependencies

If you intend to work on CyRK's code base you will want to install the following dependencies in order to run CyRK's test suite and experimental notebooks.

`conda install pytest scipy matplotlib jupyter`

`conda install` can be replaced with `pip install` if you prefer.

## Limitations and Known Issues

- [Issue 30](https://github.com/jrenaud90/CyRK/issues/30): CyRK's cysolve_ivp and pysolve_ivp does not allow for complex-valued dependent variables. 
