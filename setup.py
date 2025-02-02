from setuptools import setup, find_packages

setup(
    name='ai-content-personalization',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn',
        'numpy',
        'nltk'
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'mypy'
        ]
    },
    python_requires='>=3.8',
)