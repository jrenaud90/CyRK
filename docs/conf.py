import sys
import os
sys.path.insert(0, os.path.abspath('../CyRK'))
import toml

pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'))
with open(pyproject_path, 'r') as f:
    pyproject = toml.load(f)

project = pyproject['project']['name']
release = pyproject['project']['version']

author = 'Joe P. Renaud'

extensions = [
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
