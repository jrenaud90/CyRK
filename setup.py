from setuptools import Extension,setup

setup(
    ext_modules=[
        Extension(
            name='_cyrk.c',
            sources=['CyRK/cy/_cyrk.c'])
    ]
)
