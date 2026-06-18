from setuptools import setup, find_packages

setup(
    name="nla-steering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.47.0",
        "peft>=0.14.0",
        "accelerate>=1.2.1",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8.0",
    ],
)
