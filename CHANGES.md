# CyRK - Change Log

## 2022 - September

### v0.1.3 - Alpha

- Fixed a broken example in the readme documentation.
- Added better quick tests for both the numba and cython versions.
- Added `SciPy` dependency that was missing (required for numba integrator).
- Increased the lower limit on `numpy` package version (to fix issue [7](https://github.com/jrenaud90/CyRK/issues/7)) and removed the upper limit version.
- Removed python 3.7 as a supported version. 
- Updated graphic in readme.
- Converted over to using `pyproject.toml` instead of `setup.py`
  - removed `version.py` from project folder.

### v0.1.2 - Alpha

- Made the calling argument for the numba solver more consistent with the cython one by letting first_step==0 be
equivalent to == None

### v0.1.1 - Alpha

- Corrected issues with installation
- Improved GitHub workflows

### Initial Beta Version 0.1.0 Launched!