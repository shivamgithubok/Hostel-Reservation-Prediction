# used to project management 
# This file is used to configure the project and its dependencies.
from setuptools import setup , find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-project",
    version="0.1.0",
    author="Shivam chaudhary",
    author_email="shivam@gmail.com",
    packages=find_packages(),
    install_requires=requirements
)

# command to detect setup.py files and install the packages 
# pip install -e .