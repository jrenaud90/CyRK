import sys
import os
import CyRK
sys.path.insert(0, os.path.abspath('../CyRK'))

project = 'CyRK'
author = 'Joe P. Renaud'
release = CyRK.version

extensions = [
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
