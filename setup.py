#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

from setuptools import setup


def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with open(os.path.join(package, "__init__.py")) as f:
        return re.search("__version__ = ['\"]([^'\"]+)['\"]", f.read()).group(1)


def get_long_description():
    """
    Return the README.
    """
    with open("README.md", encoding="utf8") as f:
        return f.read()


def get_packages(package):
    """
    Return root package and all sub-packages.
    """
    return [
        dirpath
        for dirpath, dirnames, filenames in os.walk(package)
        if os.path.exists(os.path.join(dirpath, "__init__.py"))
    ]


setup(
    name="morcilla",
    version=get_version("morcilla"),
    python_requires=">=3.8",
    url="https://github.com/athenianco/morcilla",
    license="BSD",
    description="Performant async database support for Python.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Tom Christie",
    author_email="tom@tomchristie.com",
    packages=get_packages("morcilla"),
    package_data={"morcilla": ["py.typed"]},
    install_requires=["sqlalchemy>=1.3,<1.4.42"],
    extras_require={
        "postgresql": ["asyncpg-rkt"],
        "mysql": ["aiomysql"],
        "mysql+asyncmy": ["asyncmy"],
        "sqlite": ["aiosqlite"],
        "postgresql+aiopg": ["aiopg"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Internet :: WWW/HTTP",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    zip_safe=False,
)
