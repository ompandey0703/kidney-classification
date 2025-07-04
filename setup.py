#setup.py: # This file is used to package the project for distribution.
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

AUTH_USERNAME = "ompandey0703"
REPO_NAME = "kidney-classification"
AUTHOR_EMAIL = "ompandey0704@gmail.com"
setuptools.setup(
    name=f"{REPO_NAME}",
    version="0.0.1",
    author="Your Name",
    author_email=AUTHOR_EMAIL,
    description="A CNN classifier for image classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f'https://github.com/{AUTH_USERNAME}/{REPO_NAME}',
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTH_USERNAME}/{REPO_NAME}/issues",}
)
