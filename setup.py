# Release tutorial:
# https://packaging.python.org/tutorials/packaging-projects/
# Old version:
# pip install setuptools wheel twine
# python setup.py sdist bdist_wheel
# New version:
# python -m build
# Then run the following to upload to PyPI
# python -m twine upload --repository testpypi dist/*
# pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple PSID --upgrade
# python -m twine upload --repository pypi dist/*


import setuptools, os

# Get the directory of the setup.py file
dir_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(dir_path)

# requirements_file_path = os.path.join(base_dir, 'requirements.txt')
# with open(requirements_file_path, "r", encoding="utf-8") as fh:
#     requirements = fh.read().split('\n')

readme_file_path = os.path.join(base_dir, 'README.md')
with open(readme_file_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PSID",
    version="1.2.4",
    author="Omid Sani",
    author_email="omidsani@gmail.com",
    description="Python implementation for preferential subspace identification (PSID)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShanechiLab/PyPSID",
    packages=setuptools.find_packages(where='source'),
    package_dir={"": "source"},
    package_data={"PSID": ["*.mat"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    # install_requires=requirements,
)