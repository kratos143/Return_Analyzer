from setuptools import setup, find_packages

setup(
    name='return_analyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for analyzing returns',
    url='https://github.com/yourusername/return_analyzer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)