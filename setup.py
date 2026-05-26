#! /usr/bin/python3
import re
import setuptools

# Read metadata from sbmltoodejax/__init__.py as text so the build does
# not need to import the package (which would pull in jax, libsbml,
# equinox at build time and break isolated builds).
with open("sbmltoodejax/__init__.py") as f:
    init_src = f.read()


def _read_dunder(name: str) -> str:
    match = re.search(
        rf'^{name}\s*=\s*["\'](.+?)["\']', init_src, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"Could not find {name} in sbmltoodejax/__init__.py")
    return match.group(1)


__version__ = _read_dunder("__version__")
__author__ = _read_dunder("__author__")
__email__ = _read_dunder("__email__")
__url__ = _read_dunder("__url__")
__license__ = _read_dunder("__license__")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sbmltoodejax',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='lightweight library that allows to automatically parse and convert SBML models into python models written end-to-end in JAX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    license=__license__,
    packages=['sbmltoodejax'],
    install_requires=['sbmltoodepy', 'jax[cpu]', 'equinox'],
    extras_require={
        'dev': ['pytest'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
