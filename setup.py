import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

_version_ = '0.1.0'

setuptools.setup(
   name='cnnClassifier',
   version=_version_,
   author='Archi',
   author_email='archishaw622@gmail.com',
   description="A small python package for CNN app",
   long_description=long_description,
   long_description_content_type="text/markdown",
   package_dir={"": "src"},
   packages=setuptools.find_packages(where="src")
)