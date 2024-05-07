from setuptools import find_packages, setup

setup(
    name='myMLlib',
    packages=find_packages(),
    install_requires=['numpy'],
    requires=['numpy'],
    version='0.2.0',
    description='My first Python ML library',
    author='duncanchiu409',
)