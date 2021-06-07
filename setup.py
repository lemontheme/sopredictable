from setuptools import setup, find_packages
from pathlib import Path


SRC_ROOT = Path(__file__).parent / "src"
ABOUT_MODULE = SRC_ROOT / "sopredictable/about.py"

install_requires = [
    "typing-extensions;python_version < '3.8'",
]

extras_require = {
    "serve": ["fastapi"],
    "dev": ["pytest"]
}

with ABOUT_MODULE.open("r") as f:
    mod_globals = {}
    exec(f.read(), mod_globals)

version = mod_globals["__version__"]
del mod_globals


setup(
    name="sopredictable",
    version=version,
    author="Adriaan Lemmens (lemontheme)",
    author_email="lemontheme@gmail.com",
    url="https://github.com/lemontheme/sopredictable",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
