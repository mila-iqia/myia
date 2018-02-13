"""Installation script."""
from os import path

from setuptools import setup, find_packages

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='myia',
    version='0.1a',
    description='A linear algebra compiler for a subset of Python',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/mila-udem/myia',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[],
    extras_require={
        'test': ['flake8', 'pytest', 'codecov',
                 'pytest-cov', 'mypy', 'pydocstyle'],
    }
)
