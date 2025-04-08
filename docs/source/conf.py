# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import os
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

import warnings
import choreo
import choreo_GUI

warnings.filterwarnings("ignore", category=RuntimeWarning) 

project = "Choreo"
author = "Gabriel Fougeron"
project_copyright = "2021, Gabriel Fougeron"
version = choreo.metadata.__version__

# sys.path.append(os.path.abspath("./_pygments"))
# from style import PythonVSMintedStyle
# pygments_style = PythonVSMintedStyle.__qualname__

language = "en"

extensions = [
    "sphinx.ext.duration"           ,
    "sphinx.ext.doctest"            ,
    "sphinx.ext.autodoc"            ,
    "sphinx.ext.viewcode"           ,
    # "sphinx.ext.linkcode"           ,
    "sphinx.ext.todo"               ,
    "sphinx.ext.autosummary"        ,
    "sphinx.ext.mathjax"            ,
    "sphinx.ext.napoleon"           ,
    "sphinx.ext.intersphinx"        ,
    "sphinx.ext.githubpages"        ,
    "sphinx_gallery.gen_gallery"    ,
    "sphinx_needs"                  ,
    "sphinxcontrib.test_reports"    ,
    "sphinxcontrib.plantuml"        ,
    "myst_parser"                   ,
    "sphinxext.rediraffe"           ,
    "sphinxcontrib.bibtex"          ,
    # "sphinx-prompt"                 , # Incompatible versions
    "sphinxcontrib.autoprogram"     ,
    "sphinx_copybutton"             ,
    "sphinx_design"                 ,
]

todo_include_todos = True

# The suffix of source filenames.
source_suffix = ".rst"

master_doc = 'index'

# The encoding of source files.
source_encoding = "utf-8"

add_module_names = False

autodoc_typehints = "description"
autosummary_imported_members = True
autosummary_generate = True
templates_path = ['_templates']
autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'show-inheritance']

intersphinx_mapping = {
    "python"        : ("https://docs.python.org/3"                      , None),
    "sphinx"        : ("https://www.sphinx-doc.org/en/master/"          , None),
    'numpy'         : ('https://numpy.org/doc/stable/'                  , None),
    "scipy"         : ("http://docs.scipy.org/doc/scipy/reference/"     , None),
    "matplotlib"    : ("https://matplotlib.org/stable/"                 , None), 
    "networkx"      : ("https://networkx.org/documentation/stable/"     , None), 
    'pytest'        : ('https://pytest.org/en/stable/'                  , None),
    'pyquickbench'  : ('https://gabrielfougeron.github.io/pyquickbench/', None),
    'mpmath'        : ('https://mpmath.org/doc/current/'                , None),
    # 'Pillow'        : ('https://pillow.readthedocs.io/en/stable/'       , None),
    # 'cycler'        : ('https://matplotlib.org/cycler/'                 , None),
    # 'dateutil'      : ('https://dateutil.readthedocs.io/en/stable/'     , None),
    # 'ipykernel'     : ('https://ipykernel.readthedocs.io/en/latest/'    , None),
    # 'pandas'        : ('https://pandas.pydata.org/pandas-docs/stable/'  , None),
    # 'tornado'       : ('https://www.tornadoweb.org/en/stable/'          , None),
    # 'xarray'        : ('https://docs.xarray.dev/en/stable/'             , None),
    # 'meson-python'  : ('https://meson-python.readthedocs.io/en/stable/' , None),
    # 'pip'           : ('https://pip.pypa.io/en/stable/'                 , None),
}

intersphinx_disabled_reftypes = ["*"]
intersphinx_cache_limit = -1
intersphinx_timeout = 1

rediraffe_redirects = {
    "gallery": "_build/auto_examples/index"             ,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_logo_abs = os.path.join(__PROJECT_ROOT__,"docs","source","_static","img","eight_icon.png")
html_logo_rel = "_static/img/eight_icon.png"
html_logo = html_logo_rel
html_favicon = html_logo_rel
html_baseurl = "https://gabrielfougeron.github.io/choreo-docs"
html_show_sourcelink = True

html_theme_options = {
    # 'navigation_depth': 4,
    # "sidebar_includehidden" : True,
    # "search_bar_text" : "Search the docs ...",
    # "search_bar_position" : "sidebar",
    # "show_nav_level" : 4 ,
    # "show_toc_level" : 4 ,
    "show_prev_next": False,
    "header_links_before_dropdown": 7,
    "use_edit_page_button": True,
    "pygments_light_style": 'tango',
    "pygments_dark_style":  'monokai',
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
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/pyquickbench/",
        #     "icon": "fa-custom fa-pypi",
        # },
        # {
        #     "name": "Anaconda",
        #     # "url": "https://anaconda.org/conda-forge/pyquickbench",
        #     "url": "https://anaconda.org/conda-forge",
        #     "icon": "fa-custom fa-anaconda",
        # },
    ],
    "logo": {
        "text": "choreo",
        "alt_text": "choreo",
    },
    # "external_links": [{"name": "GUI", "url": "https://gabrielfougeron.github.io/choreo/", "icon": html_logo,}],
    "footer_start" : "",
    "footer_end" : "",
    "secondary_sidebar_items": ["page-toc"],
}

# Add / remove things from left sidebar
html_sidebars = {
    master_doc: [],
    "**": [
            "navbar-logo"       ,
            "sidebar-nav-bs"    ,
        ],
}

html_context = {
    "display_github"    : True              , # Integrate GitHub
    "github_user"       : "gabrielfougeron" , # Username
    "github_repo"       : "choreo"          , # Repo name
    "github_version"    : "main"            , # Version
    "version"           : "main"            , # Version
    "conf_py_path"      : "docs/source/"    , # Path in the checkout to the docs root
    "doc_path"          : "docs/source/"    ,
    "default_mode"      : "light"           ,
}

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_js_files = [
    "js/custom-icon.js",
]

################# 
# Tests-reports #
#################

tr_report_template = "./test-report/test_report_template.txt"

tr_suite_id_length = 3
tr_case_id_length = 10

tr_file = ['test-file', 'testfile', 'Test-File', 'TF_', '#ffffff', 'node']
tr_suite = ['test-suite', 'testsuite', 'Test-Suite', 'TS_', '#cccccc', 'node']
tr_case = ['test-case', 'testcase', 'Test-Case', 'TC_', '#999999', 'node']

needs_extra_options = ['introduced', 'updated', 'impacts']

# sphinx-gallery configuration

sphinx_gallery_conf = {
    # path to your examples scripts
    'filename_pattern': '/'                                 ,
    'ignore_pattern': 'NOTREADY'                            ,
    'examples_dirs': "../../examples/"                      ,
    "gallery_dirs": "_build/auto_examples/"                 ,
    "subsection_order"          : ExplicitOrder([
        "../../examples/Numerical_tricks"   ,
        "../../examples/convergence"        ,
        "../../examples/benchmarks"         ,
    ])                                                      ,
    "within_subsection_order": "FileNameSortKey"            ,
    "backreferences_dir": "_build/generated"                ,
    "image_scrapers": ("matplotlib",)                       ,
    "default_thumb_file": html_logo_abs                     ,
    "plot_gallery": True                                    ,
    'matplotlib_animations': True                           ,
    'nested_sections':True                                  ,
    "reference_url"             : {"sphinx_gallery": None,} ,
    "min_reported_time"         : 10000                     ,
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

mathjax3_config = {
    'tex': {
        'macros': {
            "dd": r"\operatorname{\mathrm{d}}",
            "eqdef": r"\mathrel{\stackrel{\mathrm{def}}{=}}",
        }
   }
}


#####################
# Napoleon settings #
#####################

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
autodoc_typehints = "description"
autosummary_generate = True

###################
# Bibtex settings #
###################

bibtex_bibfiles = ["references.bib"]

# bibtex_default_style = 'alpha'
bibtex_default_style = 'unsrt'