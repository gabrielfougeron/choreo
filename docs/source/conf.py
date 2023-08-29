# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import os
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

from sphinx_gallery.sorting import FileNameSortKey
from sphinx_pyproject import SphinxConfig

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


project = "Choreo"
author = "Gabriel Fougeron"
project_copyright = "2021, Gabriel Fougeron"
version = '0.2.0'


# sys.path.append(os.path.abspath("./_pygments"))
# from style import PythonVSMintedStyle
# pygments_style = PythonVSMintedStyle.__qualname__


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

language = "en"


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'sphinx_needs',
    'sphinxcontrib.test_reports',
    'sphinxcontrib.plantuml',
]



# The suffix of source filenames.
source_suffix = ".rst"

master_doc = 'index'

# The encoding of source files.
source_encoding = "utf-8"

add_module_names = False

autosummary_generate = True

templates_path = ['templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_logo_abs = os.path.join(__PROJECT_ROOT__,"docs","source","_static","img","eight_icon.png")
html_logo_rel = "_static/img/eight_icon.png"
html_logo = html_logo_rel
html_favicon = html_logo_rel

html_show_sourcelink = True

html_theme_options = {
    # 'navigation_depth': -1,
    # "sidebar_includehidden" : True,
    # "search_bar_text" : "Search the docs ...",
    # "search_bar_position" : "sidebar",
    # "show_nav_level" : 0 ,
    # "show_toc_level" : 0 ,
    "show_prev_next": False,
    "header_links_before_dropdown": 7,
    "use_edit_page_button": True,
    "pygment_light_style": 'tango',
    "pygment_dark_style":  'monokai',
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gabrielfougeron/choreo",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Choreo_GUI",
            "url": "https://gabrielfougeron.github.io/choreo/",
            "icon": html_logo,
            "type": "local",
        },
    ],
    "logo": {
        "text": "choreo",
        "alt_text": "choreo",
    },
    # "external_links": [{"name": "GUI", "url": "https://gabrielfougeron.github.io/choreo/", "icon": html_logo,}],
    "footer_start" : "",
    "footer_end" : "",
}

# Add / remove things from left sidebar
html_sidebars = {
    "**": ["sidebar-nav-bs"],
    # "**": [],
}

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "gabrielfougeron", # Username
    "github_repo": "choreo", # Repo name
    "github_version": "main", # Version
    "version": "main", # Version
    "conf_py_path": "docs/source/", # Path in the checkout to the docs root
    "default_mode": "light",
}

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

tr_report_template = "./test-report/test_report_template.txt"

# sphinx-gallery configuration

sphinx_gallery_conf = {
    # path to your examples scripts
    'filename_pattern': '/',
    'examples_dirs': "../../examples/",
    "gallery_dirs": "_build/auto_examples/",
    "within_subsection_order": FileNameSortKey,
    "backreferences_dir": "_build/generated",
    "image_scrapers": ("matplotlib",),
    "default_thumb_file": html_logo_abs,
    "plot_gallery": True,
    'matplotlib_animations': True,
    'nested_sections':True,
    # 'run_stale_examples': True, # Force rebuilding of examples even if the *py file did not change.
}


#############
# Latex PDF #
#############
latex_engine = "pdflatex"


# latex_documents = [("startdocname", "targetname", "title", "author", "theme", "toctree_only")]

latex_documents = [
    (master_doc, 'choreo.tex', 'Choreo documentation', 'Gabriel Fougeron', 'manual'),
]

latex_elements = {'preamble':r'\usepackage{xfrac}'}

latex_use_latex_multicolumn = False
latex_show_urls = "footnote"

latex_theme = "manual"
# latex_theme = "howto"

##################
# Math rendering #
##################
# 
# mathjax3_config = {
#   'loader': {'load': ['[tex]/xfrac']},
#   'tex': {'packages': {'[+]': ['xfrac']}},
# }