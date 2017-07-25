from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()


setup(
    name='GELMMnet',

    # Version:
    version='0.0.2',

    description='Generalized elastic-net linear mixed model',
    long_description=readme,

    # The project's main homepage.
    url='https://github.com/b-schubert/GELMMnet.git',

    # Author details
    author='Benjamin Schubert',
    author_email='benjamin_schubert@hms.harvard.edu',

    # Choose your license
    license='GPL-3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Biologists, Computational Biologists, Developer',
        'Topic :: Biostatistics :: GWAS',

        # The license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License',


        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='Generalized elastic-net linear mixed model',

    # Specify  packages via find_packages() and exclude the tests and
    # documentation:
    packages=find_packages(),


    install_requires=[
        'setuptools>=18.2', 'pandas', 'numpy', 'scipy', 'numba', 'sklearn>=0.14.1',

    ],

    dependency_links=["https://github.com/uqfoundation/pathos/tarball/master"]

)
