from setuptools import Extension,setup,numpy

setup(
    ext_modules=[
        Extension(
            name='_cyrk.c',
            sources=['CyRK/cy/_cyrk.c'],
            include_dirs=[numpy.get_include()])
    ]
)
