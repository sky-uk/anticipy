from setuptools import setup, find_packages

__version__ = "0.0.1"

# -- Edit Start
zip_safe = False

modules = []

dependencies = [
    'pandas>=0.20.3',
    'numpy>=1.13.3',
    'scipy>=1.0.0',
]

extras_require={
'r':['rpy2>=2.8.3'],
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
