import src

import subprocess

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

def custom_command():
    subprocess.call(["pip", "install", "numpy", "cython"])
    subprocess.call(["pip", "install", "-r", "requirements.txt"])


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


setup(
    name="object_detection_leaderboard",
    description="Hugging Face leaderboard for object detection models",
    version=src.__version__,
    packages=find_packages(),
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)
