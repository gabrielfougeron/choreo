# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import os
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.append(os.path.abspath("./_pygments"))

# import style
# pygments_style = style.PythonVSMintedStyle
# pygments_style = 'style.PythonVSMintedStyle'


from sphinx_gallery.sorting import FileNameSortKey
from sphinx_pyproject import SphinxConfig

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

config = SphinxConfig(os.path.join(__PROJECT_ROOT__,"pyproject.toml"), globalns=globals())


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'sphinx_needs',
    'sphinxcontrib.test_reports',
    'sphinxcontrib.plantuml'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
# html_theme = "sphinx_book_theme"
html_theme = "pydata_sphinx_theme"
html_logo = "_static/img/eight_icon.png"

html_theme_options = {
    # "github_url": "https://github.com/gabrielfougeron/choreo",
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "pygment_light_style": 'style.PythonVSMintedStyle',
    # "pygment_dark_style":  'style.PythonVSMintedStyle',
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gabrielfougeron/choreo",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Choreo_GUI",
            "url": "https://gabrielfougeron.github.io/choreo/",
            "icon": "_static/img/eight_icon.png",
            # "icon": "fa-brands fa-github",
        },
    ],
}









html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

tr_report_template = "./test-report/test_report_template.txt"

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../gallery/sample-gallery-1', '../gallery/sample-gallery-2'],
    # path to where to save gallery generated output
    'gallery_dirs': ['_galleries/sample-gallery-1', '_galleries/sample-gallery-2'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module': ('choreo'),
}   
