import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="object-detection-label-utility",  # Replace with your own username
    version="0.0.1",
    author="Brian Ofrim",
    author_email="bofrim@ualberta.ca",
    description="A utility for labeling input data for object detection machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrianOfrim/object-detection-label-utility",
    packages=setuptools.find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires=">=3.6",
)
