"""The setup script."""

from setuptools import setup, find_packages

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
    'tqdm'
]

test_requirements = []

setup(
    author="Paul Kahlmeyer",
    author_email='paul.kahlmeyer@uni-jena.de',
    python_requires='=3.9.12',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    description="An open source python library symbolic regression based on searching the space of DAGs.",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='DAG_search',
    name='DAG_search',
    packages=find_packages(include=['DAG_search', 'DAG_search.*']),
    url='https://github.com/kahlmeyer94/DAG_search'
)