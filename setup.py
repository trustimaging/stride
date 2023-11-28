
from setuptools import setup, find_packages, Extension

import version

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
        optional_requirements += [requirement.split('/')[-1].split('#')[0]]
    else:
        optional_requirements += [requirement]

setup(
    name='stride',
    version=version.__version__,
    description='A (somewhat) general optimisation framework for ultrasound medical imaging',
    long_description='A (somewhat) general optimisation framework for ultrasound medical imaging',
    url='https://github.com/trustimaging/stride',
    author='TRUST',
    author_email='c.cueto@imperial.ac.uk',
    license='AGPL-3.0',
    python_requires=">=3.8",
    packages=find_packages(exclude=['docs', 'tests', 'legacy*']),
    package_data={},
    include_package_data=True,
    install_requires=requirements,
    extras_require={'extras': optional_requirements},
    dependency_links=links,
    ext_modules=[
        Extension('_profile',
                  sources=['mosaic/profile/_profile.c'])
    ],
    scripts=[
        'mosaic/cli/mrun'
    ],
    entry_points={
        'console_scripts': [
            'mrun=mosaic.cli.mrun:go',
            'mrun_=mosaic.cli.mrun:go',
            'mscript=mosaic.cli.mscript:go',
            'mprof=mosaic.cli.mprof:go',
            'findomp=mosaic.cli.findomp:go',
        ]
    },
    zip_safe=False,
    test_suite='tests'
)
