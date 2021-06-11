#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

from sphinx.ext import apidoc

# sys.path.insert(0, os.path.abspath('.'))


regexp = re.compile(r'.*__version__ = [\'\"](.*?)[\'\"]', re.S)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
pkg_root = os.path.join(repo_root, 'src', 'muscle_synergies')
init_file = os.path.join(pkg_root, '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()
    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError(
            'Cannot find __version__ in {}'.format(init_file))


# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Muscle Synergies'
copyright = '2021, Elvis Sikora'
author = 'Elvis Sikora'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = version
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# If true, links to the reST sources are added to the pages.
#
html_show_sourcelink = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#
html_show_copyright = False

# Suppress nonlocal image warnings
suppress_warnings = ['image.nonlocal_uri']


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'description': 'Determine muscle synergies on the data outputted by the Vicon Nexus machine.',
    'show_powered_by': False,
    # 'logo': 'my-logo.png',
    'logo_name': False,
    'page_width': '80%',
}

# Custom sidebar templates, maps document names to template names.
#
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
    ]
}


# -- Custom config to work around readthedocs.org #1139 -------------------

def run_apidoc(_):
    output_path = os.path.join(repo_root, 'docs', 'source', 'api')
    apidoc.main(['-o', output_path, '-f', pkg_root])


def setup(app):
    app.connect('builder-inited', run_apidoc)
