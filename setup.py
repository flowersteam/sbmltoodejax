#! /usr/bin/python3
import setuptools
import sbmltoodejax

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sbmltoodejax',
    version=sbmltoodejax.__version__,
    author=sbmltoodejax.__author__,
    author_email=sbmltoodejax.__email__,
    description='lightweight library that allows to automatically parse and convert SBML models into python models written end-to-end in JAX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=sbmltoodejax.__url__,
    license=sbmltoodejax.__license__,
    packages=['sbmltoodejax'],
    install_requires=['sbmltoodepy', 'jax[cpu]', 'equinox'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

