from setuptools import find_packages, setup

setup(
    name='text-complexity-analyzer-cm',
    author='Hans Matos Rios',
    author_email='hans.matos@pucp.edu.pe',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'spacy',
        'Pyphen',
        'scikit-learn'
    ],
    packages=['model'],
    package_dir={'model': 'text_complexity_analyzer_cm/model'},
    package_data={'model': ['*.pkl']}
)

