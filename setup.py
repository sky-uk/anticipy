from setuptools import setup, find_packages

__version__ = "0.0.2"

# -- Edit Start
zip_safe = False

modules = []

dependencies = [
    'matplotlib==2.2.3;python_version<"3.5"', # Last version compatible with python 2.7
    'matplotlib>=2.2.3;python_version>="3.5"', # Last version compatible with python 2.7
    'numpy>=1.15.1',
    'pandas>=0.23.0',
    'scipy>=1.0.0',
]

extras_require={
}

dependency_links = [
]

entry_points = {
    "console_scripts": ['forecast = nsa.forecast.app:main']
}
# -- Edit Stop


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="anticipy",
    version=__version__,
    description="Forecasting tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pedro Capelastegui",
    author_email="pedro.capelastegui@sky.uk",
    url="https://github.com/sky-uk/anticipy",
    packages=find_packages(),
    py_modules=modules,
    include_package_data=True,
    install_requires=dependencies,
    dependency_links=dependency_links,
    zip_safe=zip_safe,
    entry_points=entry_points
)
