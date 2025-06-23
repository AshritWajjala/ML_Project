from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    Purpose: Returns list[str] of packages required for developing or deploying application.
    Input: File path to the requirements.txt
    Output: List of strings containing all the necessary packages.
    '''
    HYPHEN_E_DOT = '-e .' # Used to install setup.py actomatically
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Ashrit',
    author_email='ashritw2000@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)