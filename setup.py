from setuptools import find_packages, setup  # type: ignore

setup(
    name="fam",
    packages=find_packages(".", exclude=["tests"]),
)
