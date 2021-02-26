
from setuptools import setup, find_packages

import stride

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('requirements-optional.txt') as f:
    optionals = f.read().splitlines()

requirements = []
links = []
for requirement in required:
    if requirement[0:3] == 'git':
        links += [requirement + '#egg=' + requirement.split('/')[-1] + '-0']
        requirements += [requirement.split('/')[-1]]
    else:
        requirements += [requirement]

optional_requirements = []
optional_links = []
for requirement in optionals:
    if requirement[0:3] == 'git':
        optional_links += [requirement + '#egg=' + requirement.split('/')[-1] + '-0']
        optional_requirements += [requirement.split('/')[-1]]
    else:
        optional_requirements += [requirement]

setup(
    name='stride',
    version=stride.__version__,
    description='A (somewhat) general optimisation framework for ultrasound medical imaging',
    long_description='A (somewhat) general optimisation framework for ultrasound medical imaging',
    url='https://github.com/trustimaging/stride',
    author='TRUST',
    author_email='c.cueto@imperial.ac.uk',
    license='',
    python_requires=">=3.7",
    packages=find_packages(exclude=['docs', 'tests', 'examples']),
    package_data={},
    include_package_data=True,
    install_requires=requirements,
    extras_require={'extras': optional_requirements},
    dependency_links=links,
    entry_points='''
        [console_scripts]
        mrun=mosaic.cli.mrun:go
        mscript=mosaic.cli.mscript:go
    ''',
    zip_safe=False,
    test_suite='tests'
)
