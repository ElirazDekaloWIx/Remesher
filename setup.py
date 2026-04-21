from setuptools import setup, find_packages

setup(
    name="remesher",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "trimesh[easy]>=4.0.0",
        "pymeshlab>=2023.12",
        "numpy",
        "pygltflib",
        "pillow",
        "tqdm",
        "click",
        "flask>=3.0",
        "waitress>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "remesher=remesher.cli:main",
        ],
    },
    python_requires=">=3.10",
)
