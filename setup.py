"""The setup script."""

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'matplotlib',
    'networkx',
    'numpy',
    'scikit_learn',
    'scipy',
    'stopit',
    'sympy',
    'torch',
    'tqdm',
    'zss'
]



setup(
    name='DAG_search',
    version=1.0,
    description = "An open source python library for optimization based on searching the space of small expression DAGs.",
    long_description = readme,
    license='MIT',
    author="Paul Kahlmeyer",
    author_email='paul.kahlmeyer@uni-jena.de',
    url = 'https://github.com/kahlmeyer94/UDFS',
    packages = ['DAG_search'],
    install_requires = requirements
)
