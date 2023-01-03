import os

import numpy as np
from setuptools import Extension, setup

# Cython extensions require a setup.py in addition to pyproject.toml in order to create platform-specific wheels.
setup(
    ext_modules=[
        Extension(
            name='CyRK.cy.cyrk',
            sources=['CyRK/cy/_cyrk.pyx'],
            include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()])
    ]
)
