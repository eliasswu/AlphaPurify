from setuptools import setup, find_packages

setup(
    name="AlphaPurify",
    version="1.0.5",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "polars",
        "duckdb",
        "plotly",
        "numpy",
        "scipy",
        "pyarrow",
        "joblib",
        "scikit-learn",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful quantitative factor cleaning and analysis library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eliasswu/AlphaPurify",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)