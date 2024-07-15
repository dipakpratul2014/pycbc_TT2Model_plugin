#!/usr/bin/env python
"""
setup.py file for pycbc waveform plugin package to use TaylorT2 Model waveforms
"""

from setuptools import Extension, setup, Command
from setuptools import find_packages
from os import path

VERSION = '1.0'

def get_long_description():
    """Finds the README and reads in the description"""
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description

setup (
    name = 'pycbc-TT2Model-plugin',
    version = VERSION,
    description = 'Waveform plugin for PyCBC',
    long_description = get_long_description(),
    author = 'Pratul',
    author_email = 'dipakpratul2014@gmail.com',
    url = 'http://www.pycbc.org/',
    download_url = 'https://github.com/dipakpratul2014/pycbc_TT2Model_plugin',
    keywords = ['pycbc', 'signal processing', 'gravitational waves'],
    install_requires = ['pycbc'],
    py_modules = ['TT2_Model'],
    packages = find_packages(),
    entry_points={
        "pycbc.waveform.td": [
            "TT2Model = TT2_Model:TaylorT2_Model"
        ],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
