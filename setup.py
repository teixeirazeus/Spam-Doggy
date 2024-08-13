import re
from setuptools import setup, find_packages

with open("spam_doggy/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE
    ).group(1)

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="spam-doggy",
    version=version,
    packages=find_packages(),
    url="https://github.com/teixeirazeus/Spam-Doggy",
    project_urls={
        "Issue tracker": "https://github.com/teixeirazeus/Spam-Doggy/issues"
    },
    license="MIT",
    author="teixeirazeus",
    description="Spam Doggy is a spam classifier that uses a Naive Bayes classifier to classify emails as spam or not spam.",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["pandas", "scikit-learn", "joblib"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
)