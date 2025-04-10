from setuptools import setup, find_packages

setup(
    name="memory_rl",
    version="0.2.0",
    description="A POMDP-based simulation of mouse behavior in multi-port experiments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jercog-team/MemoryRL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
    entry_points={
        'console_scripts': [
            'run_mouse_pomdp=main:main'
        ],
    },
)
