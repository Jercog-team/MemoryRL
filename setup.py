from setuptools import setup, find_packages

setup(
    name="memoryrl",
    version="0.2.0",
    description="A POMDP-based simulation of mouse behavior in multi-port experiments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jercog-team/MemoryRL",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,  # Ensures package data is included
    package_data={
        'distributions': ['*.pkl'],  # Include all .pkl files in the data directory
    },
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
