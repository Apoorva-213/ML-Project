from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('-e')]

setup(
    name="mlproject",
    version='0.0.1',
    author='Apoorva',
    author_email='apoorva.biet@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt')
)
