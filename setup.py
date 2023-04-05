from setuptools import setup, find_packages

version = {}
with open("statcan_dialogue_dataset/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="statcan-dialogue-dataset",
    version=version["__version__"],
    author="Xing Han Lu, Siva Reddy, Harm De Vries",
    author_email="statcan.dialogue.dataset@mila.quebec",
    url="https://github.com/McGill-NLP/statcan-dialogue-dataset",
    description="The Statcan Dialogue Dataset",
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
)
