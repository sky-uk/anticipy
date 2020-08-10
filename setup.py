from setuptools import setup, find_packages

__version__ = "0.1.8"

# -- Edit Start
zip_safe = False

modules = []

dependencies = [
    'numpy>=1.15.1',
    'pandas>=0.23.0',
    'scipy>=1.0.0',
    'plotly>=3.5.0'
]

extras_require = {
    'extras': ['matplotlib>=2.2.3', 'ipython>=5.8.0',
               'notebook>=5.3', 'ipywidgets==7.5']
}

dependency_links = [
]

entry_points = {
    "console_scripts": ['forecast = nsa.forecast.app:main']
}
# -- Edit Stop


with open("PKG-INFO", "r") as fh:
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
    license="BSD",
    packages=find_packages(),
    py_modules=modules,
    include_package_data=True,
    install_requires=dependencies,
    dependency_links=dependency_links,
    zip_safe=zip_safe,
    entry_points=entry_points,
    classifiers=["License :: OSI Approved :: BSD License"],
    extras_require=extras_require
)
