from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    

    Args:
        file_path (str): _description_

    Returns:
        List[str]: _description_
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip("\n") for req in requirements]
        requirements = [req for req in requirements if req != "-e ."]

        return requirements

setup(
    name="data_science_project",
    version="0.0.1",
    author="Asaduzzaman",
    author_email="asaduzzamansheikh777@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)