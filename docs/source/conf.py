import os
import sys
sys.path.insert(0, os.path.abspath('../promptx'))

project = 'PromptX'
copyright = '2025, Connery Chen, Yihan Wang, and Bing Zhang'
author = 'Connery Chen, Yihan Wang, and Bing Zhang'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
