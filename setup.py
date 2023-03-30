#! /usr/bin/python3
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sbmltoodejax',
    version='0.2',
    author='Mayalen Etcheverry',
    author_email='mayalen.etcheverry@inria.fr',
    description='python software built upon jax, that allows to convert SBML models into python classes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/flowersteam/autodiscjax',
    license='MIT',
    packages=['sbmltoodejax'],
    install_requires=['sbmltoodepy', 'jax[cpu]', 'equinox'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

