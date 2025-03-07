from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qtlh-framework",
    version="0.1.0",
    author="QTL-H Development Team",
    author_email="contact@qtlh-framework.org",
    description="A quantum-enhanced topological linguistic hyperdimensional framework for genomic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qtlh-framework/qtlh-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "pennylane>=0.19.0",
        "qiskit>=0.34.0",
        "tensorflow>=2.7.0",
        "transformers>=4.15.0",
        "scikit-learn>=0.24.0",
        "gudhi>=3.5.0",
        "dionysus>=2.0.0",
        "gtda>=0.5.0",
        "lightning>=2.0.0",
        "wandb>=0.12.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.12b0",
            "flake8>=4.0.1",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/qtlh-framework/qtlh-framework/issues",
        "Source": "https://github.com/qtlh-framework/qtlh-framework",
        "Documentation": "https://qtlh-framework.readthedocs.io/",
    },
)
