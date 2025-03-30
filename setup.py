
from setuptools import setup, find_packages

setup(
    name='bistro',
    version='1.0.20221230',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'patsy',
        'xarray',
        'pymc',
        'bambi',
        'arviz',
        'scikit-learn',
        'graphviz',
        'seaborn',
        'tabulate',
        'matplotlib',
        # Include any other dependencies your code uses
    ],
    author='Dustin M. Burt',
    description='Bayesian inference simplification for regression of treatment on outcome',
    url='https://github.com/dmburt/bistro',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        # other classifiers if needed
    ],
)
