import src

import subprocess

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="object_detection_leaderboard",
    description="Hugging Face leaderboard for object detection models",
    version=src.__version__,
    install_requires=required,
    packages=find_packages(),
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)
