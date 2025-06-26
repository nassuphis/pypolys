from setuptools import setup, find_packages

setup(
    name='pypolys',
    version='0.1',
    packages=['polys'],
    package_data={'polys': ['test/*.py']},
    install_requires=[],  # add dependencies as needed
)

