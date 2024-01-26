from setuptools import setup, find_packages

setup(
    name='farodit',
    version='1.0',
    packages=find_packages(),
    python_requires=">=3.9, <3.12",
    install_requires=[
        'numpy>=1.23.5, <1.24',
        'tensorflow>=2.15.0, <2.16',
        'scikit-learn>=1.2.1, <1.3',
        'pandas>=1.5.3, <1.5.4',
        'matplotlib>=3.4.3, <4',
    ],
    url='https://github.com/SPbU-AI-Center/farodit',
    license='Apache-2.0',
    author='SPbU-AI-Center',
    author_email='st023633@student.spbu.ru',
    description='Framework for Analysis and Prediction on Data in Tables'
)
