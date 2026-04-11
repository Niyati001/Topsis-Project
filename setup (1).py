from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Topsis-Niyati-102303356",
    version="1.0.0",
    author="Niyati",
    author_email="niyati@example.com",
    description="A Python package for TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) multi-criteria decision analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/topsis",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        'console_scripts': [
            'topsis=topsis.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    keywords="topsis mcdm decision-making multi-criteria",
)
