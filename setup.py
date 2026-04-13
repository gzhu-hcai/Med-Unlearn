#!/usr/bin/env python
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medu",
    version="1.0.0",
    author="Minghui Huang",
    author_email="2112433114@e.gzhu.edu.cn",
    description="GRIN+: Towards Fast Yet Effective Machine Unlearning for Imbalanced Medical Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HMH4Luckhaha/Med-Unlearn",
    project_urls={
        "Bug Tracker": "https://github.com/HMH4Luckhaha/Med-Unlearn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["medu"],
    python_requires=">=3.10",
)
