from setuptools import setup, find_packages

version = {}
with open("statcan_dialogue_dataset/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="statcan-dialogue-dataset",
    version=version["__version__"],
    author="Xing Han Lu, Siva Reddy, Harm de Vries",
    author_email="statcan.dialogue.dataset@mila.quebec",
    url="https://github.com/McGill-NLP/statcan-dialogue-dataset",
    description="The Statcan Dialogue Dataset",
    long_description=long_description,
    packages=find_packages(include=["statcan_dialogue_dataset*"]),
    package_data={
        "statcan_dialogue_dataset": ["_data/*.json", "_data/members.zip"]
    },
    install_requires=[
        "pandas"
    ],
    extras_require={
        "dev": ["black", "wheel"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    # Cast long description to markdown
    long_description_content_type="text/markdown",
)
