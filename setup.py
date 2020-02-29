import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "rt_ergodic_control",
    version = "0.0.1",
    description = "real time ergodic control library",
    long_description = long_description,
    packages = setuptools.find_packages(),
)
