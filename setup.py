from setuptools import setup, find_packages

setup(
    name='fed-octtp',
    version='1.0.0',
    description='Fed-OCTTP: Federated Ocular-Calibrated Test-Time Personalization for Cross-Site Multi-Label Ocular Disease Recognition',
    author='Siyu Wang, Yanhan Hu, Zhi-Ri Tang, Tian-Tian Zhang, Yanhua Chen, Di Wang, Xu Wang',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.2.0',
        'pillow>=8.0.0',
        'tqdm>=4.50.0',
        'matplotlib>=3.3.0',
        'pyyaml>=5.4.0',
    ],
)
