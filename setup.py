import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="AccelerometerProcessing",
    version="0.0.1",
    author="Kyle Weber",
    author_email="kyle.weber@uwaterloo.ca",
    description="GENEActiv accelerometer processing for activity intensity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyleweber13/AccelerometerProcessing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6, <3.9',
    install_requires=['scipy', 'matplotlib', 'numpy', 'pandas>=1.0.4', 'pyEDFlib'],
)