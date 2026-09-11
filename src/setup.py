from setuptools import find_packages, setup
 
setup(
    name="FuncFlows",
    version="0.0.1",
    description="Normalizing flows on functional spaces",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pytest",
        "matplotlib",
        "tqdm"
        ],
    extras_require={"test": ["pytest"]},
)
 