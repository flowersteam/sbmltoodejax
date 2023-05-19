# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sbmltoodejax


# -- Project information -----------------------------------------------------

project = 'SBMLtoODEjax'
author = 'Mayalen Etcheverry'
copyright = sbmltoodejax.__copyright__
version = sbmltoodejax.__version__
release = sbmltoodejax.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
]

master_doc = "index"

myst_enable_extensions = ["dollarmath", "colon_fence"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------


html_theme = "sphinx_book_theme"
html_title = "sbmltoodejax"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/flowersteam/sbmltoodejax",
    "repository_branch": "main",
    "launch_buttons": {
        #"binderhub_url": "https://mybinder.org",
        #"notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    #"use_edit_page_button": True,
    #"use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://sbmltoodejax.readthedocs.io/en/latest/"
nb_execution_mode = "off"
nb_execution_timeout = -1