from setuptools import find_packages, setup

install_requires = []

with open('requirements.txt', 'r') as requirements:
    install_requires = [line for line in requirements]

setup(
    name='text-complexity-analyzer-cm',
    author='Hans Matos Rios',
    author_email='hans.matos@pucp.edu.pe',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    package_data={'': ['*.pkl']}
)

