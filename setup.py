from os import path

from setuptools import setup, find_namespace_packages

_long_description = open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf8').read()


def _load_requirements(filename):
    req = []
    with open(filename) as fr:
        for line in fr.readlines():
            if '#' in line:
                line = line[:line.index('#')]
            if '-f' in line:
                continue
            line = line.strip()
            if not line:
                continue
            req += [line]
    return req


setup(
    name='Vision',
    description='Vision Segmentation and Classification',
    long_description=_long_description,
    author='Daniel Dagnino',
    author_email='ddagnino@gmail.com',
    classifiers=[
        'Development Status :: Beta',
        'Intended Audience :: Job Interviewer',
        'Topic :: Software Development :: Build Tools',
        'License :: Proprietary',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    python_requires='>=3.10.4',
    install_requires=[],
    extras_require={
        'pytest': _load_requirements('requirements.txt'),
        'train': _load_requirements('requirements.txt'),
        'infer': _load_requirements('requirements.txt'),
    },
    keywords='Vision Segmentation Classification',
    package_dir={"": "."},
    packages=find_namespace_packages(where="."),
    # use_scm_version=True,
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'node-and-date',
        "relative_to": __file__,
        "root": "..",
    },
    # setup_requires=['setuptools_scm'],
    entry_points={
        'console_scripts': [
            'train = vision.cli.train:main',
            'inference = vision.cli.inference:main',
        ]
    },
    package_data={'': ['*/*.json']},
    include_package_data=True
)
