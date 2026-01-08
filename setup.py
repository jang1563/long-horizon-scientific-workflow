#!/usr/bin/env python3
"""
Setup script for Long-Horizon Scientific Workflow Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="long-horizon-scientific-workflow",
    version="1.0.0",
    author="JangKeun Kim",
    author_email="jang1563@example.com",
    description="A framework for executing multi-stage scientific discovery workflows with reasoning traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jang1563/long-horizon-scientific-workflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-workflow=src.workflow_engine:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
    },
)
